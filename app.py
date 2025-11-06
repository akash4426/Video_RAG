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
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
# 2. Load CLIP model
# --------------------------------
@st.cache_resource
def load_clip_model():
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Failed to load CLIP model: {str(e)}")
        return None, None

clip_model, processor = load_clip_model()

# --------------------------------
# 3. Extract Frames from Video
# --------------------------------
def extract_frames(video_path, sample_fps=1):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # fallback fps
            
        frame_interval = max(1, int(round(video_fps / sample_fps)))
        
        frames, frame_ids, timestamps = [], [], []
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
        return [], [], []

# --------------------------------
# 4. Get Frame Embeddings
# --------------------------------
def get_embeddings(frames, batch_size=8):
    try:
        all_embeddings = []
        progress_bar = st.progress(0, text="🔄 Generating frame embeddings...")
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu().numpy())
            progress = min((i + batch_size) / len(frames), 1.0)
            progress_bar.progress(progress, text=f"Embedding frames... {int(progress*100)}%")
        
        progress_bar.empty()
        embeddings = np.vstack(all_embeddings).astype("float32")
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# --------------------------------
# 5. Build FAISS Index
# --------------------------------
def build_index(embeddings):
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

# --------------------------------
# 6. Query and Retrieve Frames
# --------------------------------
def retrieve_frames(query, index, frames, timestamps, top_k=3):
    text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs).cpu().numpy()
    faiss.normalize_L2(text_emb)
    D, I = index.search(text_emb, top_k)
    results = [frames[i] for i in I[0]]
    result_timestamps = [timestamps[i] for i in I[0]]
    return results, D[0], result_timestamps

# --------------------------------
# 7. Gemini Summary
# --------------------------------
def get_gemini_summary(query, retrieved_frames, result_timestamps):
    try:
        # Use Streamlit secrets instead of env variables
        api_key = st.secrets["GEMINI_API_KEY"]
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision')  # Updated model name
        pil_images = [Image.fromarray(frame) for frame in retrieved_frames]

        prompt_parts = [
            f"You are a video analysis assistant. The user searched for: '{query}'.\n"
            f"The following {len(pil_images)} frames were retrieved at timestamps (s): "
            f"{', '.join([f'{t:.2f}' for t in result_timestamps])}.\n\n"
            "Based only on these frames, summarize what is happening in one paragraph."
        ] + pil_images

        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

# --------------------------------
# 8. Streamlit UI
# --------------------------------
def get_clip_segments(video_path, frame_indices, timestamps, window_size=2):
    """Extract video clips around the matched frames"""
    try:
        video = VideoFileClip(video_path)
        clips = []
        
        for timestamp in timestamps:
            start_time = max(0, timestamp - window_size/2)
            end_time = min(video.duration, timestamp + window_size/2)
            clip = video.subclip(start_time, end_time)
            clips.append(clip)
        
        video.close()
        return clips
    except Exception as e:
        st.error(f"Error extracting clips: {str(e)}")
        return []

def save_clip_as_bytes(clip):
    """Convert video clip to bytes for Streamlit display"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        clip.write_videofile(temp_file.name, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
        with open(temp_file.name, 'rb') as f:
            video_bytes = f.read()
        os.unlink(temp_file.name)
        return video_bytes
    except Exception as e:
        st.error(f"Error saving clip: {str(e)}")
        return None

def main():
    st.title("🎥 Video RAG: Semantic Search + AI Summary")
    st.write("Use natural language to search within videos using **CLIP + FAISS**, and summarize scenes using **Gemini**.")

    uploaded_file = st.file_uploader("📁 Upload a video file", type=["mp4", "mov", "avi"])
    query = st.text_input("📝 Enter your search query", "a person walking")
    sample_fps = st.slider("🎞️ Sampling FPS", 0.5, 5.0, 1.0)
    clip_duration = st.slider("🎬 Clip Duration (seconds)", 1.0, 5.0, 2.0)

    if uploaded_file and query:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        try:
            with st.spinner("⏳ Processing video..."):
                start_time = time.time()
                frames, frame_ids, timestamps = extract_frames(video_path, sample_fps)
                
                if frames:
                    embeddings = get_embeddings(frames)
                    if embeddings is not None:
                        index = build_index(embeddings)
                        results, scores, result_timestamps = retrieve_frames(query, index, frames, timestamps, top_k=3)
                        end_time = time.time()

                        st.success(f"✅ Found {len(results)} relevant clips in {end_time - start_time:.2f}s.")

                        # Get video clips
                        clips = get_clip_segments(video_path, frame_ids, result_timestamps, window_size=clip_duration)
                        
                        # Display clips and frames side by side
                        for idx, (clip, img, ts, score) in enumerate(zip(clips, results, result_timestamps, scores)):
                            st.subheader(f"Match {idx+1} | Timestamp: {ts:.2f}s | Score: {score:.4f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(Image.fromarray(img), caption="Matched Frame")
                            with col2:
                                video_bytes = save_clip_as_bytes(clip)
                                if video_bytes:
                                    st.video(video_bytes)

                        with st.expander("🧠 AI Summary (Gemini)"):
                            if st.button("Generate Summary"):
                                with st.spinner("🤖 Summarizing with Gemini..."):
                                    summary = get_gemini_summary(query, results, result_timestamps)
                                    st.write(summary)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
            # Cleanup any remaining clips
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass

if __name__ == "__main__":
    main()
