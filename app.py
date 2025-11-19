import os
import io
import cv2
import time
import math
import json
import base64
import hashlib
import logging
import tempfile
from typing import List, Tuple, Optional, Dict

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

from moviepy.editor import VideoFileClip, concatenate_videoclips

import google.generativeai as genai


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("VideoRAG")


# =========================
# Config
# =========================
CACHE_ROOT = os.path.join(tempfile.gettempdir(), "videorag_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

DEFAULT_MODEL = "openai/clip-vit-base-patch32"   # safe size for cloud
# Optional larger model (heavier). Uncomment to expose in UI choices.
MODEL_CHOICES = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]

GEMINI_MODEL = "gemini-2.5-flash"  # multimodal, fast


# =========================
# Device
# =========================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


# =========================
# Utilities
# =========================
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_hms(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0, seconds)))


# =========================
# Model loading (cached)
# =========================
@st.cache_resource
def load_clip(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


# =========================
# Video metadata & random access
# =========================
def read_video_meta(video_path: str) -> Tuple[float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, width, height


def compute_frame_schedule(video_path: str, sample_fps: float) -> Tuple[List[int], List[float]]:
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


def read_frames_by_ids(video_path: str, frame_ids: List[int]) -> List[np.ndarray]:
    """Random-access read RGB frames for given IDs."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    cap.release()
    return frames


# =========================
# Embeddings (streaming, cached to disk)
# =========================
def cache_paths(root: str, key: str) -> Dict[str, str]:
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
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def build_or_load_embeddings(
    video_path: str,
    model_name: str,
    sample_fps: float,
    batch_size: int = 16,
) -> Tuple[np.ndarray, List[int], List[float], faiss.Index, str]:
    """
    Returns: (embeddings float32 [N, D], frame_ids, timestamps, faiss_index, cache_key)
    """
    # Unique cache key per video + settings + model
    vid_hash = sha256_file(video_path)
    cache_key = f"{vid_hash}_{model_name.replace('/', '-')}_fps{sample_fps}"
    paths = cache_paths(CACHE_ROOT, cache_key)

    # If everything persisted, load
    if all(os.path.exists(p) for p in [paths["emb"], paths["ts"], paths["ids"], paths["index"], paths["meta"]]):
        logger.info("Loading cached embeddings + index from disk")
        embeddings = np.load(paths["emb"])
        timestamps = np.load(paths["ts"]).tolist()
        frame_ids = np.load(paths["ids"]).tolist()
        index = load_index(paths["index"])
        return embeddings, frame_ids, timestamps, index, cache_key

    # Else compute embeddings streaming
    st.info("⚙️ First-time processing: computing embeddings (cached for future runs).")
    frame_ids, timestamps = compute_frame_schedule(video_path, sample_fps)

    clip_model, processor = load_clip(model_name)

    all_chunks = []
    D = None

    # Progress UI
    pbar = st.progress(0.0, text="🔄 Computing embeddings…")
    total = len(frame_ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = frame_ids[start:end]
        batch_frames = read_frames_by_ids(video_path, batch_ids)
        # Clean Nones if any read failed
        valid_pairs = [(fid, fr) for fid, fr in zip(batch_ids, batch_frames) if fr is not None]
        if not valid_pairs:
            pbar.progress(end / total)
            continue

        _, frames_batch = zip(*valid_pairs)

        inputs = processor(images=list(frames_batch), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            feats = feats.detach().cpu().numpy().astype("float32")
            all_chunks.append(feats)
            if D is None:
                D = feats.shape[1]

        pbar.progress(end / total, text=f"🔄 Computing embeddings… {int((end/total)*100)}%")

    pbar.empty()

    if not all_chunks:
        raise RuntimeError("Failed to compute any embeddings for the video.")

    embeddings = np.vstack(all_chunks).astype("float32")

    # Build FAISS Index (cosine via inner-product + L2 normalization)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Persist to disk
    np.save(paths["emb"], embeddings)
    np.save(paths["ts"], np.array(timestamps, dtype=np.float32))
    np.save(paths["ids"], np.array(frame_ids, dtype=np.int64))
    save_index(index, paths["index"])
    with open(paths["meta"], "w") as f:
        json.dump(
            {
                "video_path": os.path.basename(video_path),
                "model_name": model_name,
                "sample_fps": sample_fps,
                "num_vectors": len(embeddings),
                "created_at": time.time(),
            },
            f,
        )

    return embeddings, frame_ids, timestamps, index, cache_key


# =========================
# Retrieval
# =========================
def text_search(
    query: str,
    index: faiss.Index,
    model_name: str,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    clip_model, processor = load_clip(model_name)
    text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs).cpu().numpy().astype("float32")
    faiss.normalize_L2(text_emb)
    D, I = index.search(text_emb, top_k)
    return D[0], I[0]


# =========================
# Gemini summary
# =========================
def gemini_summary(query: str, PIL_images: List[Image.Image], timestamps: List[float]) -> str:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return "⚠️ Missing Gemini API key in .streamlit/secrets.toml (GEMINI_API_KEY)."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt_parts = [
            f"You are a video analysis assistant. The user searched for: '{query}'.\n"
            f"The following {len(PIL_images)} frames were retrieved at timestamps (s): "
            f"{', '.join([f'{t:.2f}' for t in timestamps])}.\n\n"
            "First, briefly set the context. Then summarize what is happening across these frames in one concise paragraph."
        ] + PIL_images

        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"


# =========================
# Storyboard (contact sheet)
# =========================
def make_storyboard(images: List[Image.Image], cols: int = 3, cell: int = 256) -> Optional[Image.Image]:
    if not images:
        return None
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        thumb = img.copy()
        thumb.thumbnail((cell, cell))
        x, y = c * cell, r * cell
        sheet.paste(thumb, (x, y))

    return sheet


# =========================
# Clip extraction (also merged highlights)
# =========================
def extract_clips_moviepy(video_path: str, timestamps: List[float], window: float) -> Tuple[List[bytes], Optional[bytes]]:
    """Returns: list of individual clip bytes and an optional merged highlights bytes."""
    try:
        video = VideoFileClip(video_path)
        clips = []
        subclips = []

        for ts in timestamps:
            start = max(0, ts - window / 2)
            end = min(video.duration, ts + window / 2)
            sc = video.subclip(start, end)
            subclips.append(sc)

            # Write each to temp, collect bytes
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            sc.write_videofile(
                tmp_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                logger=None,
            )
            with open(tmp_path, "rb") as f:
                clips.append(f.read())
            os.remove(tmp_path)

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

        video.close()
        # Close subclips
        for sc in subclips:
            sc.close()

        return clips, merged_bytes
    except Exception as e:
        logger.exception("Clip extraction failed: %s", e)
        st.warning("⚠️ Could not extract video clips (ffmpeg/moviepy issue). Showing frames only.")
        return [], None


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Video RAG: Semantic Search + AI Summary", page_icon="🎥", layout="wide")
    st.title("🎥 Video RAG: Semantic Search + AI Summary")
    st.caption("Semantic video search with **CLIP + FAISS** and **Gemini** summaries. Caches embeddings & index for speed.")

    with st.sidebar:
        st.info(f"⚙️ Using device: **{str(device).upper()}**")
        model_name = st.selectbox("CLIP Model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
        sample_fps = st.slider("🎞️ Sampling FPS", 0.5, 6.0, 1.0, 0.5)
        batch_size = st.slider("🧪 Embedding Batch Size", 4, 64, 16, 4)
        clip_duration = st.slider("🎬 Clip Duration (seconds)", 1.0, 6.0, 2.0, 0.5)
        top_k = st.slider("🔎 Top-K matches", 1, 9, 3, 1)

        with st.expander("Advanced"):
            st.write("Embedding & index will be persisted per video hash in a temp cache.")
            if st.button("🧹 Clear all cached indices"):
                try:
                    import shutil
                    shutil.rmtree(CACHE_ROOT)
                    ensure_dir(CACHE_ROOT)
                    st.success("Cache cleared.")
                except Exception as e:
                    st.error(f"Failed to clear cache: {e}")

    uploaded = st.file_uploader("📁 Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    query = st.text_input("📝 Enter your search query", "a person walking")

    if uploaded and query:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        try:
            fps, w, h = read_video_meta(video_path)
            st.write(f"**Video info:** {w}×{h} @ {fps:.2f} FPS")

            # Build or load cached index
            t0 = time.time()
            embeddings, frame_ids, timestamps, index, cache_key = build_or_load_embeddings(
                video_path, model_name, sample_fps, batch_size=batch_size
            )
            t1 = time.time()

            st.success(f"✅ Prepared {len(embeddings)} embeddings in {t1 - t0:.2f}s (cache key: `{cache_key[:12]}…`).")

            # Query
            D, I = text_search(query, index, model_name, top_k=top_k)
            matched_ts = [timestamps[i] for i in I]
            matched_ids = [frame_ids[i] for i in I]

            # Fetch frames only for results
            result_frames = read_frames_by_ids(video_path, matched_ids)
            pil_results = [Image.fromarray(fr) for fr in result_frames if fr is not None]

            # Display results
            st.subheader("🔎 Matches")
            for rank, (score, ts, fid, fr) in enumerate(zip(D, matched_ts, matched_ids, result_frames), start=1):
                if fr is None:
                    continue
                st.markdown(f"**#{rank}** • Timestamp: `{ts:.2f}s` ({to_hms(ts)}) • Score: `{float(score):.4f}` • FrameID: `{fid}`")
                st.image(fr, caption=f"Frame @ {ts:.2f}s", use_container_width=True)

            # Storyboard & merged highlights
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Storyboard (contact sheet)")
                sheet = make_storyboard(pil_results, cols=min(3, len(pil_results)) or 1, cell=256)
                if sheet:
                    st.image(sheet, caption="Storyboard of retrieved frames", use_container_width=True)
                else:
                    st.info("No frames to compose.")

            with col2:
                st.subheader("🎬 Merged Highlights (optional)")
                if st.button("Build merged highlights"):
                    with st.spinner("Creating highlight reel…"):
                        clips, merged = extract_clips_moviepy(video_path, matched_ts, window=clip_duration)
                    if merged:
                        st.video(merged)
                        st.caption("Merged highlights of matched segments.")
                    else:
                        st.info("Merged clip not available (ffmpeg/moviepy issue).")

            # Individual clips
            with st.expander("🎞️ Individual matched clips"):
                clips, _ = extract_clips_moviepy(video_path, matched_ts, window=clip_duration)
                for idx, (clip_bytes, ts) in enumerate(zip(clips, matched_ts), start=1):
                    st.markdown(f"**Clip #{idx}** • {ts:.2f}s → {to_hms(ts)}")
                    st.video(clip_bytes)

            # Gemini summary
            with st.expander("🧠 AI Summary"):
                if st.button("Generate summary with Gemini"):
                    with st.spinner("🤖 Summarizing…"):
                        summary = gemini_summary(query, pil_results, matched_ts)
                    st.write(summary or "No summary.")

        except Exception as e:
            logger.exception("Unhandled error")
            st.error(f"❌ An error occurred: {e}")
        finally:
            if os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception:
                    pass

    else:
        st.info("Upload a video and enter a query to begin.")


if __name__ == "__main__":
    main()
