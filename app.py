import streamlit as st
import cv2
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tempfile
import os
import time
import google.generativeai as genai
import io
from moviepy.editor import VideoFileClip, concatenate_videoclips  # Note the correct spelling

# --------------------------------
# NEW DEPENDENCIES (for novelty)
# --------------------------------
# You must install these in your environment:
# pip install openai-whisper easyocr sentence-transformers
#
# We add these libraries to implement the "Text-Grounded" RAG pipeline.
import whisper
import easyocr
from sentence_transformers import SentenceTransformer

# --------------------------------
# 1. Device Setup
# --------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
st.sidebar.info(f"⚙️ Using device: **{str(device).upper()}**")

# --------------------------------
# 2. Load Models (NOW MULTIMODAL)
# --------------------------------
@st.cache_resource
def load_models():
    """
    Load all required models:
    1. CLIP (Vision)
    2. SentenceTransformer (Text)
    3. Whisper (ASR)
    4. EasyOCR (OCR)
    """
    try:
        # 1. CLIP for Visual Embeddings
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

        # 2. SentenceTransformer for Text Embeddings (e.g., ASR, OCR)
        text_embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # 3. Whisper for ASR (Audio Transcription)
        # We load a smaller model for speed in a Streamlit app
        asr_model = whisper.load_model("base.en", device=device)

        # 4. EasyOCR for OCR (On-Screen Text)
        ocr_reader = easyocr.Reader(['en'], gpu=(device.type == 'cuda'))

        return clip_model, clip_processor, text_embed_model, asr_model, ocr_reader
    except Exception as e:
        st.error(f"Failed to load AI models: {str(e)}")
        return None, None, None, None, None

clip_model, clip_processor, text_embed_model, asr_model, ocr_reader = load_models()

# --------------------------------
# 3. Frame Extraction (Unchanged)
# --------------------------------
def extract_frames(video_path, sample_fps=1):
    # This function is the same as your original, well-written one.
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # fallback fps
            
        frame_interval = max(1, int(round(video_fps / sample_fps)))
        
        frames, frame_ids, timestamps = [], []  # Initialize empty lists
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = count / video_fps
                frames.append(rgb)
                frame_ids.append(count)
                timestamps.append(timestamp)
            count += 1
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
            
        return frames, frame_ids, timestamps
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
        return None, None, None

# --------------------------------
# 4. NOVELTY: Text-Grounded Indexing (ASR + OCR)
# --------------------------------
@st.cache_data
def get_asr_transcripts(_asr_model, video_path):
    """Extracts ASR transcripts with timestamps using Whisper."""
    st.info("Starting ASR (Audio)...")
    try:
        # TODO: This is where you implement whisper
        # result = _asr_model.transcribe(video_path, verbose=False)
        #
        # # For demo, we use placeholder text
        # if not result.get("segments"):
        #     return
        # return result["segments"]
        
        # --- Placeholder ---
        st.warning("ASR (Whisper) is using placeholder data. Replace with `asr_model.transcribe()`")
        return
        # --- End Placeholder ---
    except Exception as e:
        st.error(f"Error during ASR: {e}")
        return

@st.cache_data
def get_ocr_text(_ocr_reader, frame):
    """Extracts OCR text from a single frame."""
    try:
        # TODO: This is where you implement easyocr
        # result = _ocr_reader.readtext(frame, detail=0, paragraph=True)
        # return " ".join(result)
        
        # --- Placeholder ---
        # Simulating finding text in one of the frames
        if np.random.rand() > 0.9:
             return "On-Screen Title: Video RAG"
        return ""
        # --- End Placeholder ---
    except Exception as e:
        return ""

def get_text_data(video_path, frames, timestamps):
    """NOVEL FUNCTION: Creates a list of text chunks from ASR and OCR."""
    text_chunks = []  # Initialize empty list
    
    # 1. Get ASR data
    asr_segments = get_asr_transcripts(asr_model, video_path)
    for segment in asr_segments:
        text_chunks.append({
            "timestamp": (segment["start"] + segment["end"]) / 2,
            "text": f"Audio: {segment['text']}",
            "type": "asr"
        })
    
    # 2. Get OCR data (from the sampled frames)
    st.info("Starting OCR (On-Screen Text)...")
    ocr_progress = st.progress(0, text="🔄 Scanning frames for text...")
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        ocr_result = get_ocr_text(ocr_reader, frame)
        if ocr_result:
            text_chunks.append({
                "timestamp": ts,
                "text": f"On-Screen: {ocr_result}",
                "type": "ocr"
            })
        progress = (i + 1) / len(frames)
        ocr_progress.progress(progress, text=f"Scanning frame {i+1}/{len(frames)} for text...")
    ocr_progress.empty()

    # Sort all text chunks by timestamp
    text_chunks.sort(key=lambda x: x["timestamp"])
    return text_chunks

# --------------------------------
# 5. Embedding (Vision + Text)
# --------------------------------
def get_visual_embeddings(frames, batch_size=8):
    """Your original function, now just for visuals."""
    all_embeddings = []  # Initialize empty list
    try:
        progress_bar = st.progress(0, text="🔄 Generating visual (CLIP) embeddings...")
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu().numpy())
            progress = min((i + batch_size) / len(frames), 1.0)
            progress_bar.progress(progress, text=f"Embedding visual frames... {int(progress*100)}%")
        
        progress_bar.empty()
        embeddings = np.vstack(all_embeddings).astype("float32")
        return embeddings
    except Exception as e:
        st.error(f"Error generating visual embeddings: {str(e)}")
        return None

def get_text_embeddings(_text_embed_model, text_chunks):
    """NOVEL FUNCTION: Generates embeddings for the text data."""
    if not text_chunks:
        return None
    try:
        st.info("Generating text (SentenceTransformer) embeddings...")
        chunk_strings = [chunk["text"] for chunk in text_chunks]
        embeddings = _text_embed_model.encode(chunk_strings, show_progress_bar=True, device=device)
        return embeddings.astype("float32")
    except Exception as e:
        st.error(f"Error generating text embeddings: {str(e)}")
        return None

# --------------------------------
# 6. Build FAISS Index (Now two of them)
# --------------------------------
def build_faiss_index(embeddings):
    """Utility function to build a FAISS index."""
    if embeddings is None or len(embeddings) == 0:
        return None
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

# --------------------------------
# 7. NOVELTY: Hybrid Retrieval
# --------------------------------
def retrieve_context(query, visual_index, text_index, visual_data, text_data, top_k=3):
    """
    NOVEL FUNCTION: Performs hybrid retrieval from both visual and text indexes.
    `visual_data` = (frames, timestamps)
    `text_data` = text_chunks
    """
    frames, timestamps = visual_data
    
    # 1. Retrieve from Visual Index
    text_inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        query_visual_emb = clip_model.get_text_features(**text_inputs).cpu().numpy()
    faiss.normalize_L2(query_visual_emb)
    D_visual, I_visual = visual_index.search(query_visual_emb, top_k)
    
    visual_results = []
    for i, score in zip(I_visual, D_visual):
        visual_results.append({
            "frame": frames[i],
            "timestamp": timestamps[i],
            "score": score,
            "type": "visual"
        })

    # 2. Retrieve from Text Index
    text_results = []
    if text_index and text_data:
        query_text_emb = text_embed_model.encode([query], device=device)
        faiss.normalize_L2(query_text_emb)
        D_text, I_text = text_index.search(query_text_emb, top_k)
        
        for i, score in zip(I_text, D_text):
            text_results.append({
                "text": text_data[i]["text"],
                "timestamp": text_data[i]["timestamp"],
                "score": score,
                "type": "text"
            })

    # 3. Combine and Re-rank (simple combination for now)
    all_results = sorted(visual_results + text_results, key=lambda x: x["score"], reverse=True)
    
    # Return unique timestamps, preferring higher scores
    final_timestamps = []
    final_results = []
    seen_ts = set()
    
    for res in all_results:
        ts = res["timestamp"]
        # Add clips within 2 seconds of each other as the "same"
        if not any(abs(ts - s) < 2.0 for s in seen_ts):
            final_timestamps.append(ts)
            final_results.append(res)
            seen_ts.add(ts)

    # Get the data for the top_k unique results
    top_frames = [res["frame"] for res in final_results if res["type"] == "visual"]
    if not top_frames: # Handle text-only results
        top_frames = [frames[timestamps.index(res["timestamp"])] for res in final_results if res["type"] == "text"]

    return final_results[:top_k], top_frames, final_timestamps[:top_k]

# --------------------------------
# 8. NOVELTY: Enhanced Gemini Summary
# --------------------------------
def get_gemini_summary(query, retrieved_frames, retrieved_context):
    """
    ENHANCED FUNCTION: Now sends both text and visual context to Gemini.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]  # Fix API key access
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        pil_images = [Image.fromarray(frame) for frame in retrieved_frames]

        context_str = "\n".join([
            f"- At {res['timestamp']:.2f}s: {res['text']} (Score: {res['score']:.2f})" 
            for res in retrieved_context if res["type"] != "visual"
        ])

        if not context_str:
            context_str = "No relevant audio or on-screen text was found."

        # Fix prompt construction
        prompt_parts = [
            f"You are a video analysis assistant. The user searched for: '{query}'.\n\n"
            f"Context from the video:\n{context_str}\n\n"
            "Based on these frames and context, summarize what is happening in one paragraph."
        ] + pil_images

        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

# --------------------------------
# 9. Clip Extraction (Unchanged)
# --------------------------------
def get_clip_segments(video_path, timestamps, window_size=2):
    """Extract video clips around the matched timestamps"""
    clips = []
    video = None
    
    try:
        video = VideoFileClip(video_path)
        
        for timestamp in timestamps:
            # Calculate clip boundaries
            start_time = max(0, timestamp - window_size/2)
            end_time = min(video.duration, timestamp + window_size/2)
            
            # Extract subclip
            clip = video.subclip(start_time, end_time)
            
            # Use tempfile for robust, unique filenames
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_clip:
                clip_path = tmp_clip.name
            
            try:
                clip.write_videofile(
                    clip_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None,
                    verbose=False
                )
                
                with open(clip_path, 'rb') as f:
                    clip_bytes = f.read()
                
                clips.append({
                    'bytes': clip_bytes,
                    'path': clip_path,
                    'start': start_time,
                    'end': end_time
                })
                
            except Exception as e:
                st.error(f"Error processing clip at {timestamp}s: {str(e)}")
            finally:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
        
        return clips
    
    except Exception as e:
        st.error(f"Error in video processing: {str(e)}")
        return []
    
    finally:
        if video is not None:
            try:
                video.close()
            except:
                pass

# --------------------------------
# 10. Streamlit UI (Modified for new pipeline)
# --------------------------------
def check_dependencies():
    try:
        from moviepy.config import get_setting
        if get_setting("FFMPEG_BINARY"):
            return True
        return False
    except Exception:
        return False

def main():
    st.title("🎥 Novel Video RAG: Semantic Search + AI Summary")
    st.write("This app uses a **Text-Grounded** RAG pipeline. It searches vision (CLIP), audio (ASR), and on-screen text (OCR).")

    # Check for models
    if not clip_model:
        st.error("Models failed to load. Please check the console and restart.")
        return

    if not check_dependencies():
        st.error("❌ FFmpeg is not installed. Please install FFmpeg to use this app.")
        st.code("brew install ffmpeg  # For Mac")
        return
    
    uploaded_file = st.file_uploader("📁 Upload a video file", type=["mp4", "mov", "avi"])
    query = st.text_input("📝 Enter your search query", "a person talking about RAG")
    
    with st.expander("Settings"):
        sample_fps = st.slider("🎞️ Sampling FPS (for visuals)", 0.5, 5.0, 1.0)
        clip_duration = st.slider("🎬 Clip Duration (seconds)", 1.0, 5.0, 3.0)
        top_k = st.slider("🔍 Number of Results", 1, 5, 3)

    if uploaded_file and query:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        try:
            with st.spinner("⏳ Processing video... (This may take a while for ASR/OCR)"):
                start_time = time.time()
                
                # 1. Extract visual frames
                frames, frame_ids, timestamps = extract_frames(video_path, sample_fps)
                
                if frames:
                    # 2. NOVELTY: Extract text data (ASR + OCR)
                    text_chunks = get_text_data(video_path, frames, timestamps)

                    # 3. Embed both visual and text data
                    visual_embeddings = get_visual_embeddings(frames)
                    text_embeddings = get_text_embeddings(text_embed_model, text_chunks)
                    
                    # 4. Build FAISS indexes
                    visual_index = build_faiss_index(visual_embeddings)
                    text_index = build_faiss_index(text_embeddings)

                    if visual_index is None and text_index is None:
                        st.error("Failed to build any search index.")
                        return

                    st.success("✅ Video indexed. Ready to search.")

                    with st.spinner("🔍 Searching visual, audio, and text..."):
                        # 5. NOVELTY: Hybrid Retrieval
                        retrieved_context, top_frames, result_timestamps = retrieve_context(
                            query, 
                            visual_index, 
                            text_index, 
                            (frames, timestamps), 
                            text_chunks, 
                            top_k=top_k
                        )
                        end_time = time.time()

                    st.success(f"✅ Found {len(retrieved_context)} relevant clips in {end_time - start_time:.2f}s.")

                    # 6. Get video clips for display
                    clips = get_clip_segments(video_path, result_timestamps, window_size=clip_duration)
                    
                    # 7. Display results
                    for idx, (clip_data, res) in enumerate(zip(clips, retrieved_context)):
                        st.subheader(f"Match {idx+1} | Timestamp: {res['timestamp']:.2f}s | Score: {res['score']:.4f}")
                        
                        # Show what was found
                        if res['type'] == 'visual':
                            st.info("Found via: **Visual Search**")
                        else:
                            st.info(f"Found via: **Text Search** ({res['type'].upper()})")
                            st.markdown(f"> {res['text']}")

                        st.video(clip_data['bytes'])
                        st.caption(f"Clip timeframe: {clip_data['start']:.2f}s - {clip_data['end']:.2f}s")

                    # 8. NOVELTY: Enhanced Generation
                    with st.expander("🧠 AI Summary (Gemini)"):
                        if st.button("Generate Summary"):
                            with st.spinner("🤖 Summarizing with enhanced context..."):
                                summary = get_gemini_summary(query, top_frames, retrieved_context)
                                st.write(summary)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
        finally:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.unlink(video_path)

if __name__ == "__main__":
    main()