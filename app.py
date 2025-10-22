import streamlit as st
import cv2
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tempfile
import os
import time
from tqdm import tqdm
import google.generativeai as genai  # Added for Gemini

# --------------------------------
# 1. Device Setup
# --------------------------------
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# 2. Load CLIP model
# --------------------------------
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

clip_model, processor = load_clip_model()

# --------------------------------
# 3. Extract Frames from Video
# --------------------------------
def extract_frames(video_path, sample_fps=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
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
    return frames, frame_ids, timestamps

# --------------------------------
# 4. Get Frame Embeddings
# --------------------------------
def get_embeddings(frames, batch_size=8):
    all_embeddings = []
    # Add a progress bar for embedding generation
    progress_bar = st.progress(0, text="Generating frame embeddings...")
    for i in tqdm(range(0, len(frames), batch_size), desc="Embedding frames"):
        batch = frames[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
        # Update progress bar
        progress_bar.progress((i + batch_size) / len(frames), text="Generating frame embeddings...")
    
    progress_bar.empty() # Clear progress bar
    embeddings = np.vstack(all_embeddings).astype("float32")
    return embeddings

# --------------------------------
# 5. Build FAISS Index
# --------------------------------
def build_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
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
# 7. NEW: Get Gemini Summary
# --------------------------------
def get_gemini_summary(query, retrieved_frames, result_timestamps):
    """
    Generates a summary from Gemini using the query and retrieved image frames.
    """
    try:
        # Use gemini-2.5-flash for multimodal input
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        
        # Convert numpy arrays (frames) to PIL Images
        pil_images = [Image.fromarray(frame) for frame in retrieved_frames]
        
        # Create the prompt parts
        prompt_parts = [
            f"You are a video analysis assistant. The user searched for: '{query}'.\n"
            f"The following {len(pil_images)} frames were retrieved from the video at these timestamps (in seconds): {', '.join([f'{t:.2f}s' for t in result_timestamps])}.\n\n"
            "Based *only* on these retrieved frames, please provide a brief, one-paragraph summary of what is happening."
        ]
        
        # Add images to the prompt
        for img in pil_images:
            prompt_parts.append(img)
            
        # Generate content
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        # Handle potential API key errors or other issues
        if "API_KEY_INVALID" in str(e):
            return "Error: Invalid Gemini API Key. Please check the key and try again."
        return f"Error calling Gemini: {e}"

# --------------------------------
# 8. Streamlit UI
# --------------------------------
st.title("🎥 Video-based RAG System")
st.write("Retrieve **relevant frames** from a video using natural language queries (powered by CLIP + FAISS) and summarize with Gemini.")

# Add Gemini API Key input
gemini_api_key = st.text_input("Enter your Gemini API Key (optional, for summary)", type="password")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
query = st.text_input("Enter your search query", "a person walking")
sample_fps = st.slider("Sampling FPS", 0.5, 5.0, 1.0)

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.info("⏳ Processing video, building embeddings, and retrieving frames... please wait.")
    start_time = time.time()
    
    frames, frame_ids, timestamps = extract_frames(video_path, sample_fps)
    
    if not frames:
        st.error("No frames extracted. Please check the video file or sampling rate.")
    else:
        embeddings = get_embeddings(frames)
        index = build_index(embeddings)
        results, scores, result_timestamps = retrieve_frames(query, index, frames, timestamps, top_k=3)
        end_time = time.time()

        st.success(f"✅ Retrieval complete! Found {len(results)} frames in {end_time - start_time:.2f} seconds.")

        # Display retrieved frames
        cols = st.columns(len(results) or 1)
        for idx, (img, ts, score) in enumerate(zip(results, result_timestamps, scores)):
            with cols[idx]:
                st.image(Image.fromarray(img), caption=f"Rank {idx+1} | Time: {ts:.2f}s | Score: {score:.4f}")

        # --- MODIFIED: Gemini Summary Expander ---
        with st.expander("🧠 AI Summary (Gemini)"):
            if not gemini_api_key:
                st.info("Please enter your Gemini API Key above to enable AI summaries.")
            else:
                # Button to trigger summary generation
                if st.button("Generate Summary"):
                    try:
                        # Configure the genai library with the user's key
                        genai.configure(api_key=gemini_api_key)
                        
                        with st.spinner("Asking Gemini to summarize the scenes..."):
                            summary = get_gemini_summary(query, results, result_timestamps)
                            st.write(summary)
                    
                    except Exception as e:
                        st.error(f"Failed to configure Gemini. Check your API key. Error: {e}")

    # Clean up temp file
    if os.path.exists(video_path):
        os.unlink(video_path)
