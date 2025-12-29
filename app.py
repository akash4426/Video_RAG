"""
Video RAG: Semantic Video Search & AI Summarization
====================================================

A Retrieval-Augmented Generation (RAG) system for semantic video search using:
- CLIP (OpenAI) for multimodal embeddings
- FAISS (Facebook AI) for efficient similarity search
- Gemini (Google) for AI-powered summaries
- MoviePy for video clip extraction
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import csv
import io
import os
import time
import math
import json
import base64
import hashlib
import logging
import tempfile
from typing import List, Tuple, Optional, Dict

# Third-party imports for numerical computing and UI
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Deep learning and embeddings
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss  # Similarity search engine

# Video processing
from moviepy.editor import VideoFileClip, concatenate_videoclips

# AI summarization
import google.generativeai as genai

# Metrics for research evaluation
from metrics import RetrievalMetrics, format_metrics_for_display


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("VideoRAG")


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Cache directory for storing embeddings and FAISS indices
# Using system temp directory ensures compatibility across platforms
CACHE_ROOT = os.path.join(tempfile.gettempdir(), "videorag_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

# Default CLIP model (balance between performance and resource usage)
DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# Available CLIP model variants
# - base-patch32: Faster, less accurate (recommended for cloud deployment)
# - large-patch14: Slower, more accurate (requires more GPU memory)
MODEL_CHOICES = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]

# Gemini model for AI summarization
# gemini-1.5-flash: Fast multimodal model with vision capabilities
GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def get_device() -> torch.device:
    """
    Automatically detect and return the best available compute device.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - Apple Silicon (M1/M2/M3)
    2. CUDA - NVIDIA GPUs
    3. CPU - Fallback for any system
    
    Returns:
        torch.device: The optimal device for tensor operations
    
    Example:
        >>> device = get_device()
        >>> print(device)  # cuda:0, mps, or cpu
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sha256_file(path: str) -> str:
    """
    Compute SHA-256 hash of a file for cache key generation.
    
    Uses streaming approach to handle large video files efficiently
    without loading entire file into memory.
    
    Args:
        path: Path to the file
    
    Returns:
        str: Hexadecimal hash string (64 characters)
    
    Example:
        >>> hash_val = sha256_file("video.mp4")
        >>> print(hash_val)  # 'a1b2c3d4...'
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 1MB chunks to handle large files
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    
    Note:
        exist_ok=True prevents errors if directory already exists
    """
    os.makedirs(path, exist_ok=True)


def to_hms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    
    Example:
        >>> to_hms(3665.5)
        '01:01:05'
    """
    return time.strftime("%H:%M:%S", time.gmtime(max(0, seconds)))


# ============================================================================
# MODEL LOADING (WITH STREAMLIT CACHING)
# ============================================================================

@st.cache_resource
def load_clip(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model and processor with caching.
    
    The @st.cache_resource decorator ensures:
    - Model is loaded only once per app session
    - Shared across all users
    - Persists across reruns
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Tuple of (CLIPModel, CLIPProcessor)
    
    Architecture:
        - Model: Neural network for generating embeddings
        - Processor: Handles image/text preprocessing
    
    Example:
        >>> model, processor = load_clip("openai/clip-vit-base-patch32")
        >>> model.eval()  # Set to evaluation mode
    """
    # Load model with safetensors (faster, safer serialization format)
    model = CLIPModel.from_pretrained(
        model_name, 
        use_safetensors=True
    ).to(device)
    
    # Load processor for input preprocessing
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    return model, processor


# ============================================================================
# VIDEO METADATA & FRAME EXTRACTION
# ============================================================================

def read_video_meta(video_path: str) -> Tuple[float, int, int]:
    """
    Extract video metadata using MoviePy (better codec support).
    """
    try:
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(video_path)
        fps = video.fps or 30.0
        width, height = video.size
        video.close()
        
        logger.info(f"Video metadata (MoviePy): {width}x{height} @ {fps} FPS")
        return fps, width, height
        
    except Exception as e:
        logger.warning(f"MoviePy failed to read metadata: {str(e)}")
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        
        cap.release()
        logger.info(f"Video metadata (OpenCV): {width}x{height} @ {fps} FPS")
        return fps, width, height


def compute_frame_schedule(
    video_path: str, 
    sample_fps: float
) -> Tuple[List[int], List[float]]:
    """
    Compute which frames to extract based on sampling rate using MoviePy.
    """
    try:
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(video_path)
        video_fps = video.fps
        duration = video.duration
        video.close()
        
        if not video_fps or video_fps <= 0:
            video_fps = 30.0
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(round(video_fps / sample_fps)))
        
        # Calculate total frames
        total_frames = int(duration * video_fps)
        
        frame_ids, timestamps = [], []
        
        # Generate frame schedule
        for fid in range(0, total_frames, frame_interval):
            timestamp = fid / video_fps
            if timestamp <= duration:  # Ensure within video duration
                frame_ids.append(fid)
                timestamps.append(timestamp)
        
        if not frame_ids:
            raise ValueError("No frames scheduled for extraction")
        
        logger.info(f"Scheduled {len(frame_ids)} frames (every {frame_interval} frames)")
        
        return frame_ids, timestamps
        
    except Exception as e:
        logger.error(f"MoviePy frame scheduling failed: {str(e)}")
        # Fallback to OpenCV
        return compute_frame_schedule_opencv(video_path, sample_fps)


def compute_frame_schedule_opencv(
    video_path: str, 
    sample_fps: float
) -> Tuple[List[int], List[float]]:
    """
    Fallback: OpenCV-based frame scheduling.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    
    frame_ids, timestamps = [], []

    for fid in range(0, total_frames, frame_interval):
        timestamp = fid / video_fps
        frame_ids.append(fid)
        timestamps.append(timestamp)

    cap.release()
    
    if not frame_ids:
        raise ValueError("No frames scheduled for extraction")
    
    return frame_ids, timestamps


def read_frames_by_ids(
    video_path: str, 
    frame_ids: List[int]
) -> List[np.ndarray]:
    """
    Random-access read of specific video frames using MoviePy (more compatible than OpenCV).
    """
    frames = []
    
    try:
        # Use MoviePy instead of OpenCV for better codec support
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(video_path)
        logger.info(f"✓ Opened video with MoviePy")
        logger.info(f"Video properties:")
        logger.info(f"  - Duration: {video.duration}s")
        logger.info(f"  - FPS: {video.fps}")
        logger.info(f"  - Size: {video.size}")
        
        total_frames = int(video.duration * video.fps)
        video_fps = video.fps
        
        if total_frames == 0 or video_fps == 0:
            logger.error("Video has 0 frames or 0 FPS")
            video.close()
            raise RuntimeError(f"Invalid video properties: {total_frames} frames @ {video_fps} FPS")
        
        successful = 0
        failed = 0
        
        for idx, fid in enumerate(frame_ids):
            # Validate frame ID
            if fid < 0 or fid >= total_frames:
                logger.warning(f"Frame ID {fid} out of range [0, {total_frames})")
                frames.append(None)
                failed += 1
                continue
            
            try:
                # Convert frame ID to timestamp
                timestamp = fid / video_fps
                
                # Get frame at timestamp
                frame = video.get_frame(timestamp)
                
                if frame is None or frame.size == 0:
                    if idx < 3:
                        logger.warning(f"Failed to read frame {fid} at {timestamp}s")
                    frames.append(None)
                    failed += 1
                    continue
                
                # MoviePy returns RGB frames (not BGR like OpenCV)
                frames.append(frame)
                successful += 1
                
            except Exception as e:
                if idx < 3:
                    logger.error(f"Exception reading frame {fid}: {str(e)}")
                frames.append(None)
                failed += 1
        
        video.close()
        
        if successful == 0:
            logger.error(f"Frame extraction completely failed: 0/{len(frame_ids)} frames read")
        else:
            logger.info(f"Frame extraction: {successful} successful, {failed} failed out of {len(frame_ids)}")
        
        return frames
        
    except Exception as e:
        logger.error(f"Failed to open video with MoviePy: {str(e)}")
        # Fallback to OpenCV method
        return read_frames_by_ids_opencv(video_path, frame_ids)


def read_frames_by_ids_opencv(
    video_path: str, 
    frame_ids: List[int]
) -> List[np.ndarray]:
    """
    Fallback: OpenCV-based frame extraction.
    """
    frames = []
    
    # Try to open video with different backends
    backends = [
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_ANY, "ANY"),
    ]
    
    cap = None
    successful_backend = None
    
    for backend, name in backends:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            successful_backend = name
            logger.info(f"✓ Opened video with OpenCV backend: {name}")
            break
        else:
            logger.warning(f"Failed to open video with OpenCV backend: {name}")
            if cap:
                cap.release()
    
    if not cap or not cap.isOpened():
        logger.error(f"Cannot open video with any OpenCV backend: {video_path}")
        raise RuntimeError(f"Cannot open video {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Decode fourcc
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    logger.info(f"Video properties (OpenCV):")
    logger.info(f"  - Backend: {successful_backend}")
    logger.info(f"  - Resolution: {width}x{height}")
    logger.info(f"  - FPS: {video_fps}")
    logger.info(f"  - Total frames: {total_frames}")
    logger.info(f"  - Codec (fourcc): {fourcc_str}")
    
    if total_frames == 0 or video_fps == 0:
        logger.error("Video has 0 frames or 0 FPS - likely unsupported codec")
        cap.release()
        raise RuntimeError(f"Invalid video properties: {total_frames} frames @ {video_fps} FPS")
    
    successful = 0
    failed = 0

    for idx, fid in enumerate(frame_ids):
        if fid < 0 or fid >= total_frames:
            logger.warning(f"Frame ID {fid} out of range [0, {total_frames})")
            frames.append(None)
            failed += 1
            continue
            
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            
            if not ret or frame is None:
                if idx < 3:
                    logger.warning(f"Failed to read frame {fid}")
                frames.append(None)
                failed += 1
                continue
            
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(f"Invalid frame shape at {fid}: {frame.shape}")
                frames.append(None)
                failed += 1
                continue
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            successful += 1
            
        except Exception as e:
            if idx < 3:
                logger.error(f"Exception reading frame {fid}: {str(e)}")
            frames.append(None)
            failed += 1

    cap.release()
    
    if successful == 0:
        logger.error(f"OpenCV frame extraction failed: 0/{len(frame_ids)} frames")
        logger.error(f"Video codec '{fourcc_str}' may not be supported")
    else:
        logger.info(f"OpenCV extraction: {successful} successful, {failed} failed")
    
    return frames


# ============================================================================
# EMBEDDING GENERATION & CACHING
# ============================================================================

def cache_paths(root: str, key: str) -> Dict[str, str]:
    """
    Generate file paths for cached data.
    
    Cache structure:
        cache_root/
        └── {video_hash}_{model}_{fps}/
            ├── embeddings.npy      # Float32 array [N, 512]
            ├── timestamps.npy      # Float32 array [N]
            ├── frame_ids.npy       # Int64 array [N]
            ├── index.faiss         # FAISS index file
            └── meta.json           # Metadata (model, fps, etc.)
    
    Args:
        root: Cache root directory
        key: Unique cache key (video hash + settings)
    
    Returns:
        Dictionary mapping logical names to file paths
    """
    base = os.path.join(root, key)
    ensure_dir(base)
    
    return {
        "base": base,
        "emb": os.path.join(base, "embeddings.npy"),
        "ts": os.path.join(base, "timestamps.npy"),
        "ids": os.path.join(base, "frame_ids.npy"),
        "index": os.path.join(base, "index.faiss"),
        "meta": os.path.join(base, "meta.json"),
    }


def save_index(index: faiss.Index, path: str) -> None:
    """
    Persist FAISS index to disk.
    
    Args:
        index: FAISS index object
        path: Output file path
    """
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    """
    Load FAISS index from disk.
    
    Args:
        path: Path to saved index file
    
    Returns:
        Loaded FAISS index
    """
    return faiss.read_index(path)


def build_or_load_embeddings(
    video_path: str,
    model_name: str,
    sample_fps: float,
    batch_size: int = 16,
) -> Tuple[np.ndarray, List[int], List[float], faiss.Index, str]:
    """
    Core function: Build embeddings or load from cache with detailed error tracking.
    """
    # Generate unique cache key
    vid_hash = sha256_file(video_path)
    cache_key = f"{vid_hash}_{model_name.replace('/', '-')}_fps{sample_fps}"
    paths = cache_paths(CACHE_ROOT, cache_key)

    # Check if all cache files exist
    required_files = [
        paths["emb"], 
        paths["ts"], 
        paths["ids"], 
        paths["index"], 
        paths["meta"]
    ]
    
    if all(os.path.exists(p) for p in required_files):
        # Cache hit: Load from disk
        logger.info("Loading cached embeddings + index from disk")
        embeddings = np.load(paths["emb"])
        timestamps = np.load(paths["ts"]).tolist()
        frame_ids = np.load(paths["ids"]).tolist()
        index = load_index(paths["index"])
        return embeddings, frame_ids, timestamps, index, cache_key

    # Cache miss: Compute embeddings
    st.info("⚙️ First-time processing: computing embeddings (cached for future runs).")
    
    # Step 1: Determine which frames to extract
    try:
        frame_ids, timestamps = compute_frame_schedule(video_path, sample_fps)
        logger.info(f"✓ Scheduled {len(frame_ids)} frames for extraction")
        st.info(f"📊 Processing {len(frame_ids)} frames from video...")
    except Exception as e:
        error_msg = f"Failed to compute frame schedule: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise RuntimeError(error_msg)

    # Step 2: Load CLIP model
    try:
        clip_model, processor = load_clip(model_name)
        logger.info(f"✓ Loaded CLIP model: {model_name}")
    except Exception as e:
        error_msg = f"Failed to load CLIP model: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise RuntimeError(error_msg)

    # Step 3: Process frames in batches
    all_chunks = []
    D = None  # Embedding dimension (will be set on first batch)
    failed_batches = 0
    successful_frames = 0
    total_frames_attempted = 0

    # Progress tracking
    pbar = st.progress(0.0, text="🔄 Computing embeddings…")
    total = len(frame_ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = frame_ids[start:end]
        total_frames_attempted += len(batch_ids)
        
        try:
            # Read frames for this batch
            batch_frames = read_frames_by_ids(video_path, batch_ids)
            
            # Filter out failed reads (None values)
            valid_pairs = [
                (fid, fr) for fid, fr in zip(batch_ids, batch_frames) 
                if fr is not None
            ]
            
            if not valid_pairs:
                failed_batches += 1
                logger.warning(f"Batch {start}-{end}: No valid frames (all {len(batch_ids)} frames failed)")
                pbar.progress(end / total, text=f"🔄 Processing… {end}/{total} frames (⚠️ {failed_batches} empty batches)")
                continue

            _, frames_batch = zip(*valid_pairs)
            successful_frames += len(frames_batch)
            
            logger.info(f"Batch {start}-{end}: {len(frames_batch)}/{len(batch_ids)} frames valid")

            # Preprocess images for CLIP
            try:
                inputs = processor(
                    images=list(frames_batch), 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
            except Exception as proc_e:
                logger.error(f"Processor failed on batch {start}-{end}: {str(proc_e)}")
                failed_batches += 1
                pbar.progress(end / total)
                continue
            
            # Generate embeddings (no gradient computation needed)
            try:
                with torch.no_grad():
                    # Extract image features using CLIP vision encoder
                    feats = clip_model.get_image_features(**inputs)
                    
                    # L2 normalization (convert to unit vectors)
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    
                    # Convert to numpy and ensure float32 (FAISS requirement)
                    feats = feats.detach().cpu().numpy().astype("float32")
                    
                    all_chunks.append(feats)
                    
                    if D is None:
                        D = feats.shape[1]  # Typically 512 for CLIP
                        logger.info(f"✓ Embedding dimension: {D}")
                        
            except Exception as model_e:
                logger.error(f"Model inference failed on batch {start}-{end}: {str(model_e)}")
                failed_batches += 1
                pbar.progress(end / total)
                continue

        except Exception as e:
            failed_batches += 1
            logger.error(f"Unexpected error processing batch {start}-{end}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        # Update progress bar
        success_rate = (successful_frames / total_frames_attempted * 100) if total_frames_attempted > 0 else 0
        pbar.progress(
            end / total, 
            text=f"🔄 Computing embeddings… {int((end/total)*100)}% | ✓ {successful_frames} frames ({success_rate:.1f}%)"
        )

    pbar.empty()

    # Log final statistics
    logger.info(f"=== Processing Complete ===")
    logger.info(f"Total frames scheduled: {len(frame_ids)}")
    logger.info(f"Total frames attempted: {total_frames_attempted}")
    logger.info(f"Successful frames: {successful_frames}")
    logger.info(f"Failed batches: {failed_batches}")
    logger.info(f"Success rate: {(successful_frames/total_frames_attempted*100):.1f}%")
    logger.info(f"Embedding chunks: {len(all_chunks)}")

    if not all_chunks:
        error_details = (
            f"❌ Failed to compute any embeddings for the video.\n\n"
            f"**Diagnostics:**\n"
            f"- Scheduled frames: {len(frame_ids)}\n"
            f"- Frames attempted: {total_frames_attempted}\n"
            f"- Successful frames: {successful_frames}\n"
            f"- Failed batches: {failed_batches}\n"
            f"- Video path: {video_path}\n"
            f"- Device: {device}\n"
            f"- Model: {model_name}\n\n"
            f"**Possible causes:**\n"
            f"1. Video file is corrupted or unreadable\n"
            f"2. Video codec not supported by OpenCV\n"
            f"3. Memory issues on device: {device}\n"
            f"4. Frame extraction failed for all frames\n\n"
            f"**Suggestions:**\n"
            f"- Try a different video file\n"
            f"- Try reducing batch size to 4\n"
            f"- Check video file can be played normally\n"
            f"- Check system logs above for specific errors"
        )
        logger.error(error_details)
        st.error(error_details)
        raise RuntimeError("No embeddings could be computed from the video.")

    # Step 4: Concatenate all batches
    try:
        embeddings = np.vstack(all_chunks).astype("float32")
        logger.info(f"✓ Created embedding matrix: {embeddings.shape}")
    except Exception as e:
        error_msg = f"Failed to concatenate embeddings: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise RuntimeError(error_msg)

    # Step 5: Build FAISS index
    try:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info(f"✓ Built FAISS index with {index.ntotal} vectors")
    except Exception as e:
        error_msg = f"Failed to build FAISS index: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise RuntimeError(error_msg)

    # Step 6: Persist to disk
    try:
        np.save(paths["emb"], embeddings)
        np.save(paths["ts"], np.array(timestamps[:len(embeddings)], dtype=np.float32))
        np.save(paths["ids"], np.array(frame_ids[:len(embeddings)], dtype=np.int64))
        save_index(index, paths["index"])
        
        # Save metadata for debugging/inspection
        with open(paths["meta"], "w") as f:
            json.dump(
                {
                    "video_path": os.path.basename(video_path),
                    "model_name": model_name,
                    "sample_fps": sample_fps,
                    "num_vectors": len(embeddings),
                    "successful_frames": successful_frames,
                    "failed_batches": failed_batches,
                    "success_rate": f"{(successful_frames/total_frames_attempted*100):.2f}%",
                    "created_at": time.time(),
                },
                f,
            )
        logger.info(f"✓ Cached embeddings to {paths['base']}")
    except Exception as e:
        logger.warning(f"Failed to save cache (non-critical): {str(e)}")
        # Don't raise - we can still return the computed embeddings

    st.success(f"✅ Successfully processed {successful_frames}/{total_frames_attempted} frames ({(successful_frames/total_frames_attempted*100):.1f}%)")
    
    return embeddings, frame_ids[:len(embeddings)], timestamps[:len(embeddings)], index, cache_key


# ============================================================================
# SIMILARITY SEARCH
# ============================================================================

def text_search(
    query: str,
    index: faiss.Index,
    model_name: str,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semantic search: Find frames matching text query.
    
    Process:
    1. Convert text query to embedding using CLIP text encoder
    2. Normalize embedding (unit vector)
    3. Search FAISS index for nearest neighbors
    4. Return top-k matches with scores
    
    Args:
        query: Natural language search query
        index: Pre-built FAISS index
        model_name: CLIP model identifier
        top_k: Number of results to return
    
    Returns:
        Tuple of (distances, indices)
        - distances: Similarity scores [top_k]
        - indices: Frame indices [top_k]
    
    Similarity metric:
        Cosine similarity = dot(query_emb, frame_emb)
        Range: [-1, 1] where 1 = identical, -1 = opposite
    
    Example:
        >>> D, I = text_search("person walking", index, model, top_k=3)
        >>> print(D)  # [0.89, 0.85, 0.82]
        >>> print(I)  # [120, 450, 780]
    """
    # Load CLIP model
    clip_model, processor = load_clip(model_name)
    
    # Preprocess text query
    text_inputs = processor(
        text=[query], 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # Generate text embedding
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb = text_emb.cpu().numpy().astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(text_emb)
    
    # Search index
    # D = distances (higher = more similar for inner product)
    # I = indices (frame IDs)
    D, I = index.search(text_emb, top_k)
    
    return D[0], I[0]


# ============================================================================
# AI SUMMARIZATION (GEMINI)
# ============================================================================

def gemini_summary(
    query: str, 
    PIL_images: List[Image.Image], 
    timestamps: List[float]
) -> str:
    """
    Generate AI summary of retrieved frames using Google's Gemini.
    
    Gemini capabilities:
        - Multimodal understanding (text + images)
        - Context-aware descriptions
        - Temporal reasoning across frames
    
    Args:
        query: Original search query
        PIL_images: Retrieved frames as PIL Images
        timestamps: Corresponding timestamps
    
    Returns:
        AI-generated summary text
    
    API Requirements:
        - GEMINI_API_KEY in Streamlit secrets
        - Internet connection
        - Google Cloud billing enabled
    
    Example:
        >>> images = [frame1, frame2, frame3]
        >>> timestamps = [45.2, 102.7, 158.3]
        >>> summary = gemini_summary("person walking", images, timestamps)
        >>> print(summary)
        # "The video shows a person in casual clothing walking through
        #  a park. The scene transitions from daylight to dusk..."
    """
    try:
        # Retrieve API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return "⚠️ Missing Gemini API key in .streamlit/secrets.toml (GEMINI_API_KEY)."

    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Construct prompt with context
        prompt_parts = [
            f"You are a video analysis assistant. The user searched for: '{query}'.\n"
            f"The following {len(PIL_images)} frames were retrieved at timestamps (s): "
            f"{', '.join([f'{t:.2f}' for t in timestamps])}.\n\n"
            "First, briefly set the context. Then summarize what is happening "
            "across these frames in one concise paragraph."
        ] + PIL_images

        # Generate content (multimodal input)
        response = model.generate_content(prompt_parts)
        return response.text
        
    except Exception as e:
        logger.exception("Gemini API error")
        return f"⚠️ Gemini Error: {str(e)}"


# ============================================================================
# VISUALIZATION: STORYBOARD (CONTACT SHEET)
# ============================================================================

def make_storyboard(
    images: List[Image.Image], 
    cols: int = 3, 
    cell: int = 256
) -> Optional[Image.Image]:
    """
    Create a contact sheet (grid) of images.
    
    Visual layout:
        ┌───┬───┬───┐
        │ 1 │ 2 │ 3 │
        ├───┼───┼───┤
        │ 4 │ 5 │ 6 │
        └───┴───┴───┘
    
    Args:
        images: List of PIL Images
        cols: Number of columns
        cell: Size of each cell (pixels)
    
    Returns:
        Composite image or None if no images
    
    Use case:
        Quick visual overview of all search results
    """
    if not images:
        return None
    
    # Calculate grid dimensions
    rows = math.ceil(len(images) / cols)
    
    # Create blank canvas (dark gray background)
    sheet = Image.new("RGB", (cols * cell, rows * cell), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    # Place each image in grid
    for idx, img in enumerate(images):
        r = idx // cols  # Row index
        c = idx % cols   # Column index
        
        # Resize to fit cell while maintaining aspect ratio
        thumb = img.copy()
        thumb.thumbnail((cell, cell))
        
        # Calculate position
        x, y = c * cell, r * cell
        
        # Paste into canvas
        sheet.paste(thumb, (x, y))

    return sheet


# ============================================================================
# VIDEO CLIP EXTRACTION
# ============================================================================

def extract_clips_moviepy(
    video_path: str, 
    timestamps: List[float], 
    window: float
) -> Tuple[List[bytes], Optional[bytes]]:
    """
    Extract video clips around matched timestamps.
    
    Process:
    1. For each timestamp, extract [timestamp - window/2, timestamp + window/2]
    2. Save individual clips as bytes
    3. Optionally merge all clips into highlight reel
    
    Args:
        video_path: Path to source video
        timestamps: List of matched timestamps
        window: Duration of each clip (seconds)
    
    Returns:
        Tuple of:
        - List of individual clip bytes
        - Merged highlight reel bytes (or None if failed)
    
    Technical details:
        - Uses H.264 codec for broad compatibility
        - AAC audio codec
        - Temporary files cleaned up after processing
    
    Example:
        >>> clips, merged = extract_clips_moviepy(
        ...     "video.mp4", 
        ...     [45.2, 102.7], 
        ...     window=2.0
        ... )
        >>> # Creates 2-second clips: [44.2-46.2s] and [101.7-103.7s]
    """
    try:
        video = VideoFileClip(video_path)
        clips = []
        subclips = []

        for ts in timestamps:
            # Calculate clip boundaries
            start = max(0, ts - window / 2)
            end = min(video.duration, ts + window / 2)
            
            # Extract subclip
            sc = video.subclip(start, end)
            subclips.append(sc)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            sc.write_videofile(
                tmp_path,
                codec="libx264",      # H.264 video codec
                audio_codec="aac",    # AAC audio codec
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                logger=None,          # Suppress moviepy logs
            )
            
            # Read clip as bytes
            with open(tmp_path, "rb") as f:
                clips.append(f.read())
            
            # Cleanup temp file
            os.remove(tmp_path)

        # Create merged highlight reel
        merged_bytes = None
        if subclips:
            merged = concatenate_videoclips(subclips, method="compose")
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpm:
                merged_path = tmpm.name
            
            merged.write_videofile(
                merged_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-merged-audio.m4a",
                remove_temp=True,
                logger=None,
            )
            
            with open(merged_path, "rb") as f:
                merged_bytes = f.read()
            
            os.remove(merged_path)

        # Cleanup
        video.close()
        for sc in subclips:
            sc.close()

        return clips, merged_bytes
        
    except Exception as e:
        logger.exception("Clip extraction failed: %s", e)
        st.warning("⚠️ Could not extract video clips (ffmpeg/moviepy issue). "
                   "Showing frames only.")
        return [], None


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """
    Main Streamlit application entry point.
    """
    
    # ========================================================================
    # PAGE CONFIGURATION
    # ========================================================================
    
    st.set_page_config(
        page_title="Video RAG: Semantic Search + AI Summary",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video RAG: Semantic Search + AI Summary")
    st.caption(
        "Semantic video search with **CLIP + FAISS** and **Gemini** summaries. "
        "Caches embeddings & index for speed."
    )

    # ========================================================================
    # SIDEBAR: CONFIGURATION OPTIONS
    # ========================================================================
    
    with st.sidebar:
        # Display current device
        st.info(f"⚙️ Using device: **{str(device).upper()}**")
        
        # Model selection
        model_name = st.selectbox(
            "CLIP Model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(DEFAULT_MODEL)
        )
        
        # Sampling rate (frames per second)
        sample_fps = st.slider(
            "🎞️ Sampling FPS",
            min_value=0.5,
            max_value=6.0,
            value=1.0,
            step=0.5,
            help="Higher FPS = more frames = better coverage but slower"
        )
        
        # Batch size for embedding computation
        batch_size = st.slider(
            "🧪 Embedding Batch Size",
            min_value=4,
            max_value=64,
            value=16,
            step=4,
            help="Higher batch size = faster but uses more memory"
        )
        
        # Clip duration around matched timestamps
        clip_duration = st.slider(
            "🎬 Clip Duration (seconds)",
            min_value=1.0,
            max_value=6.0,
            value=2.0,
            step=0.5,
            help="Duration of extracted video clips around matches"
        )
        
        # Number of search results
        top_k = st.slider(
            "🔎 Top-K matches",
            min_value=1,
            max_value=9,
            value=3,
            step=1,
            help="Number of matching frames to retrieve"
        )

        # Research Metrics Configuration
        with st.expander("📊 Research Metrics"):
            st.write("Enable comprehensive metrics for research evaluation")
            
            enable_metrics = st.checkbox(
                "Enable Metrics Calculation",
                value=False,
                help="Compute retrieval metrics (requires ground truth for full metrics)"
            )
            
            if enable_metrics:
                st.info(
                    "💡 **Tip**: Provide ground truth relevant frame IDs (comma-separated) "
                    "for complete metrics including Top-K Retrieval Percentage"
                )
                
                ground_truth_input = st.text_input(
                    "Ground Truth Frame IDs (optional)",
                    "",
                    help="Comma-separated frame IDs that are relevant to your query. "
                         "Example: 120,450,780,1200"
                )
            else:
                ground_truth_input = ""

        # Advanced options
        with st.expander("Advanced Options"):
            st.write("Embedding & index will be persisted per video hash "
                     "in a temp cache.")
            st.write(f"**Cache location:** `{CACHE_ROOT}`")
            
            # Cache clearing button
            if st.button("🧹 Clear all cached indices"):
                try:
                    import shutil
                    if os.path.exists(CACHE_ROOT):
                        shutil.rmtree(CACHE_ROOT)
                        ensure_dir(CACHE_ROOT)
                        st.success("✅ Cache cleared!")
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to clear cache: {e}")

    # ========================================================================
    # MAIN AREA: VIDEO UPLOAD & SEARCH
    # ========================================================================
    
    # File uploader with better instructions
    st.markdown("### 📹 Upload Video")
    st.info("💡 **Tip:** Use MP4 files with H.264 codec for best compatibility")
    
    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Recommended: MP4 with H.264 codec. Max size depends on your Streamlit Cloud tier."
    )
    
    # Search query input
    query = st.text_input(
        "📝 Enter your search query",
        "a person walking",
        help="Describe what you're looking for in the video"
    )

    # Process if both video and query are provided
    if uploaded and query:
        # Save uploaded file to temporary location with proper extension
        file_extension = os.path.splitext(uploaded.name)[1] or ".mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
        
        logger.info(f"Uploaded video saved to: {video_path}")
        logger.info(f"Original filename: {uploaded.name}")
        logger.info(f"File size: {os.path.getsize(video_path)} bytes")

        try:
            # ================================================================
            # STEP 1: VIDEO METADATA & VALIDATION
            # ================================================================
            
            with st.spinner("🔍 Analyzing video file..."):
                try:
                    fps, w, h = read_video_meta(video_path)
                    st.success(f"✅ Video loaded: **{w}×{h}** @ **{fps:.2f} FPS**")
                except Exception as meta_error:
                    st.error(
                        f"❌ Failed to read video metadata: {str(meta_error)}\n\n"
                        f"**Possible issues:**\n"
                        f"- Unsupported video codec\n"
                        f"- Corrupted video file\n"
                        f"- File format not recognized by OpenCV\n\n"
                        f"**Suggestion:** Try converting your video to MP4 (H.264) using:\n"
                        f"```bash\n"
                        f"ffmpeg -i input.{file_extension} -c:v libx264 -c:a aac output.mp4\n"
                        f"```"
                    )
                    raise

            # ================================================================
            # STEP 2: BUILD OR LOAD EMBEDDINGS
            # ================================================================
            
            t0 = time.time()
            embeddings, frame_ids, timestamps, index, cache_key = \
                build_or_load_embeddings(
                    video_path,
                    model_name,
                    sample_fps,
                    batch_size=batch_size
                )
            t1 = time.time()

            st.success(
                f"✅ Prepared **{len(embeddings)}** embeddings in **{t1 - t0:.2f}s** "
                f"(cache key: `{cache_key[:12]}…`)"
            )

            # ================================================================
            # STEP 3: SEARCH
            # ================================================================
            
            with st.spinner("🔍 Searching for matches..."):
                D, I = text_search(query, index, model_name, top_k=top_k)
            
            # Map indices back to timestamps and frame IDs
            matched_ts = [timestamps[i] for i in I]
            matched_ids = [frame_ids[i] for i in I]

            # Fetch frames only for matched results
            result_frames = read_frames_by_ids(video_path, matched_ids)
            pil_results = [
                Image.fromarray(fr) for fr in result_frames if fr is not None
            ]

            if not pil_results:
                st.warning("⚠️ No valid frames could be retrieved for search results.")
                return

            # ================================================================
            # STEP 3.5: COMPUTE METRICS (if enabled)
            # ================================================================
            
            metrics_results = None
            if enable_metrics:
                metrics_calculator = RetrievalMetrics()
                
                # Parse ground truth if provided
                ground_truth_indices = None
                if ground_truth_input.strip():
                    try:
                        ground_truth_indices = [
                            int(x.strip()) 
                            for x in ground_truth_input.split(',')
                            if x.strip()
                        ]
                        logger.info(f"Ground truth provided: {len(ground_truth_indices)} relevant frames")
                    except ValueError:
                        st.warning("⚠️ Invalid ground truth format. Using format: 120,450,780")
                        ground_truth_indices = None
                
                # Compute metrics
                metrics_results = metrics_calculator.evaluate_query(
                    retrieved_indices=I,
                    similarity_scores=D,
                    relevant_indices=ground_truth_indices,
                    k=top_k
                )
                
                logger.info(f"Metrics computed: {metrics_results}")

            # ================================================================
            # STEP 4: DISPLAY RESULTS
            # ================================================================
            
            st.markdown("---")
            st.subheader("🔎 Search Results")
            
            # Display metrics if enabled
            if enable_metrics and metrics_results:
                with st.expander("📊 Retrieval Metrics", expanded=True):
                    metrics_display = format_metrics_for_display(metrics_results)
                    st.markdown(metrics_display)
                    
                    # Add download buttons for metrics
                    col_json, col_csv = st.columns(2)
                    
                    with col_json:
                        if st.button("📥 Export Metrics (JSON)"):
                            metrics_json = json.dumps(metrics_results, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=metrics_json,
                                file_name=f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    with col_csv:
                        if st.button("📥 Export Metrics (CSV)"):
                            output = io.StringIO()
                            writer = csv.writer(output)
                            writer.writerow(['Metric', 'Value'])
                            for key, value in sorted(metrics_results.items()):
                                writer.writerow([key, value])
                            
                            st.download_button(
                                label="Download CSV",
                                data=output.getvalue(),
                                file_name=f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            
            st.markdown("---")
            
            for rank, (score, ts, fid, fr) in enumerate(
                zip(D, matched_ts, matched_ids, result_frames), start=1
            ):
                if fr is None:
                    continue
                
                # Display metadata
                st.markdown(
                    f"**#{rank}** • Timestamp: `{ts:.2f}s` ({to_hms(ts)}) • "
                    f"Score: `{float(score):.4f}` • FrameID: `{fid}`"
                )
                
                # Display frame
                st.image(
                    fr,
                    caption=f"Frame @ {ts:.2f}s",
                    use_container_width=True
                )

            # ================================================================
            # STEP 5: STORYBOARD & HIGHLIGHTS
            # ================================================================
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Storyboard")
                sheet = make_storyboard(
                    pil_results,
                    cols=min(3, len(pil_results)) or 1,
                    cell=256
                )
                if sheet:
                    st.image(
                        sheet,
                        caption="Contact sheet of all matches",
                        use_container_width=True
                    )
                else:
                    st.info("No frames to compose.")

            with col2:
                st.subheader("🎬 Video Highlights")
                if st.button("🎬 Create Highlight Reel"):
                    with st.spinner("Creating video clips..."):
                        clips, merged = extract_clips_moviepy(
                            video_path,
                            matched_ts,
                            window=clip_duration
                        )
                    
                    if merged:
                        st.video(merged)
                        st.caption("Merged highlights of matched segments.")
                    else:
                        st.info("Merged clip not available (ffmpeg/moviepy issue).")

            # ================================================================
            # STEP 6: INDIVIDUAL CLIPS
            # ================================================================
            
            with st.expander("🎞️ Individual Clips"):
                with st.spinner("Extracting clips..."):
                    clips, _ = extract_clips_moviepy(
                        video_path,
                        matched_ts,
                        window=clip_duration
                    )
                
                if clips:
                    for idx, (clip_bytes, ts) in enumerate(
                        zip(clips, matched_ts), start=1
                    ):
                        st.markdown(f"**Clip #{idx}** • {ts:.2f}s → {to_hms(ts)}")
                        st.video(clip_bytes)
                else:
                    st.info("No clips available.")

            # ================================================================
            # STEP 7: AI SUMMARY
            # ================================================================
            
            with st.expander("🧠 AI Summary"):
                if st.button("Generate AI Summary"):
                    with st.spinner("🤖 Generating summary with Gemini..."):
                        summary = gemini_summary(query, pil_results, matched_ts)
                    st.markdown(summary or "No summary available.")

        except Exception as e:
            logger.exception("Unhandled error in main")
            st.error(f"❌ An error occurred: {str(e)}")
            
            # Show detailed error info
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
            
        finally:
            # Cleanup: Remove temporary video file
            if 'video_path' in locals() and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                    logger.info(f"Cleaned up temp file: {video_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

    else:
        # Show instructions when no video uploaded
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Upload a video** (MP4 recommended with H.264 codec)
        2. **Enter a search query** (e.g., "person walking", "red car", "sunset")
        3. **View results** with timestamps and similarity scores
        4. **Generate highlights** and AI summaries
        
        **Supported formats:** MP4, MOV, AVI, MKV
        
        **Best results:** Use well-encoded H.264 MP4 files
        """)


if __name__ == "__main__":
    main()
