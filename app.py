# =========================================================
# 0. CLEAN STARTUP BLOCK  —  quiet logs & speed cloud runtime
# =========================================================
import os, sys, warnings, logging

# ---- Quiet noisy libraries ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"     # HuggingFace silence
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ctranslate2").setLevel(logging.ERROR)

# ---- Silence warnings globally ----
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor")
warnings.filterwarnings("ignore", message="The compute type inferred")

# ---- Optional: redirect stderr (hides all red logs) ----
# =========================================================
# CLEAN STARTUP BLOCK  —  quiet logs & speed cloud runtime
# =========================================================
import os, sys, warnings, logging

# ---- Quiet noisy libraries ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"     # HuggingFace silence
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ctranslate2").setLevel(logging.ERROR)

# ---- Silence warnings globally ----
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor")
warnings.filterwarnings("ignore", message="The compute type inferred")

# ---- Optional: redirect stderr (hides all red logs) ----
# =========================================================
# CLEAN STARTUP BLOCK  —  quiet logs & speed cloud runtime
# =========================================================
import os, sys, warnings, logging

# ---- Quiet noisy libraries ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"     # HuggingFace silence
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ctranslate2").setLevel(logging.ERROR)

# ---- Silence warnings globally ----
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor")
warnings.filterwarnings("ignore", message="The compute type inferred")

# ---- Optional: redirect stderr (hides all red logs) ----
class DevNull:
    def write(self, _): pass
    def flush(self): pass
sys.stderr = DevNull()

# ---- Fix MoviePy ImageMagick binary crash ----
import moviepy.config_defaults as cfg
cfg.IMAGEMAGICK_BINARY = "unset"
# =========================================================

# =========================================================


# =========================================================
# 1. IMPORTS
# =========================================================
import streamlit as st
import cv2
import torch
import numpy as np
import os
import time
import tempfile
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import google.generativeai as genai

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'

# Basic error handling
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

# Device setup
@handle_errors
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()

# Load CLIP model
@st.cache_resource
def load_clip_model():
    try:
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        ).to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"Failed to load CLIP model: {str(e)}")
        return None, None

# Frame extraction
@handle_errors
def extract_frames(video_path, sample_fps=1):
    frames, timestamps = [], []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return [], []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / sample_fps)
    
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                timestamps.append(count / fps)
            count += 1
    finally:
        cap.release()
        
    return frames, timestamps

# Gemini summary
def get_summary(query, frames, timestamps):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision')
        
        images = [Image.fromarray(frame) for frame in frames]
        prompt = f"Query: {query}\nDescribe what's happening in these video frames."
        
        response = model.generate_content([prompt] + images)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Main UI
def main():
    st.title("🎥 Video Search")
    st.write("Upload a video and search within it using natural language")
    
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    query = st.text_input("Enter search query", "person walking")
    fps = st.slider("Sampling FPS", 0.5, 5.0, 1.0)
    
    if uploaded_file and query:
        with st.spinner("Processing video..."):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            try:
                # Extract frames
                frames, timestamps = extract_frames(video_path, fps)
                if frames:
                    st.success(f"Extracted {len(frames)} frames")
                    
                    # Display results
                    cols = st.columns(3)
                    for idx, frame in enumerate(frames[:3]):
                        cols[idx].image(frame, caption=f"Frame at {timestamps[idx]:.2f}s")
                    
                    # Generate summary
                    if st.button("Generate Summary"):
                        summary = get_summary(query, frames[:3], timestamps[:3])
                        st.write(summary)
                        
            finally:
                # Cleanup
                if os.path.exists(video_path):
                    os.unlink(video_path)

if __name__ == "__main__":
    main()
