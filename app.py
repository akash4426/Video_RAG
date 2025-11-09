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
from PIL import Image, ImageDraw

import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

from moviepy.editor import VideoFileClip, concatenate_videoclips

import google.generativeai as genai


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("VideoRAG")


# =========================
# Config
# =========================
CACHE_ROOT = os.path.join(tempfile.gettempdir(), "videorag_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
GEMINI_MODEL = "gemini-2.5-flash"


# =========================
# Device Setup
# =========================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


# =========================
# Hashing & Cache Paths
# =========================
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# Model Loading
# =========================
@st.cache_resource
def load_clip_model(model_name: str):
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


# =========================
# Frame Extraction
# =========================
def extract_frames(video_path: str, sample_fps: float) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(round(fps / sample_fps)))
    frames, timestamps = [], []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(count / fps)
        count += 1
    cap.release()
    return frames, timestamps


# =========================
# Embedding & Indexing
# =========================
def get_embeddings(frames: List[np.ndarray], processor, clip_model, batch_size=16):
    embeddings = []
    pbar = st.progress(0.0, text="Generating frame embeddings...")
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
        pbar.progress((i + batch_size) / len(frames))
    pbar.empty()
    return np.vstack(embeddings).astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def search_frames(query: str, index, clip_model, processor, frames, timestamps, top_k=3):
    text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs).cpu().numpy()
    faiss.normalize_L2(text_emb)
    D, I = index.search(text_emb, top_k)
    results = [frames[i] for i in I[0]]
    times = [timestamps[i] for i in I[0]]
    return results, times, D[0]


# =========================
# Storyboard Utility
# =========================
def make_storyboard(images: List[Image.Image], cols=3, size=256):
    if not images:
        return None
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * size, rows * size))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = img.copy()
        thumb.thumbnail((size, size))
        sheet.paste(thumb, (c * size, r * size))
    return sheet


# =========================
# Gemini Summary (restored & enhanced)
# =========================
def get_gemini_summary(query: str, retrieved_frames: List[np.ndarray], timestamps: List[float], merged_clip: Optional[bytes] = None):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        pil_images = [Image.fromarray(f) for f in retrieved_frames]

        prompt = [
            f"You are a video understanding assistant. The user searched for: '{query}'.",
            f"The retrieved frames are at timestamps (s): {', '.join([f'{t:.2f}' for t in timestamps])}.",
            "First, describe what seems to be happening in these frames collectively.",
            "Then, summarize the key visual events in one short paragraph.",
        ]

        # If highlight clip available, add it as visual context
        if merged_clip:
            video_data = {"mime_type": "video/mp4", "data": merged_clip}
            parts = prompt + pil_images + [video_data]
        else:
            parts = prompt + pil_images

        response = model.generate_content(parts)
        return response.text or "No summary generated."
    except Exception as e:
        return f"⚠️ Gemini summarization failed: {str(e)}"


# =========================
# Video Clip Extraction
# =========================
def get_clip_segments(video_path: str, timestamps: List[float], window: float):
    clips = []
    video = VideoFileClip(video_path)
    for ts in timestamps:
        start = max(0, ts - window / 2)
        end = min(video.duration, ts + window / 2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            path = tmp.name
        clip = video.subclip(start, end)
        clip.write_videofile(path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, logger=None)
        with open(path, "rb") as f:
            clips.append(f.read())
        os.remove(path)
    video.close()

    # Merge all highlights
    subclips = [VideoFileClip(video_path).subclip(max(0, t - window / 2), min(VideoFileClip(video_path).duration, t + window / 2)) for t in timestamps]
    merged_bytes = None
    if subclips:
        merged = concatenate_videoclips(subclips, method="compose")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpm:
            merged_path = tmpm.name
        merged.write_videofile(merged_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-merged.m4a", remove_temp=True, logger=None)
        with open(merged_path, "rb") as f:
            merged_bytes = f.read()
        os.remove(merged_path)
        merged.close()
        for s in subclips:
            s.close()
    return clips, merged_bytes


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="🎥 Video RAG", layout="wide")
    st.title("🎥 Video RAG: Semantic Search + AI Video Summarization")
    st.caption("Search and summarize videos using **CLIP + FAISS + Gemini**")

    uploaded = st.file_uploader("📁 Upload a video", type=["mp4", "mov", "avi", "mkv"])
    query = st.text_input("📝 Enter your search query", "a person walking")
    sample_fps = st.slider("🎞️ Frame Sampling FPS", 0.5, 5.0, 1.0)
    clip_duration = st.slider("🎬 Clip Duration (seconds)", 1.0, 5.0, 2.0)
    top_k = st.slider("🔎 Top K Frames", 1, 9, 3)

    if uploaded and query:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        try:
            clip_model, processor = load_clip_model(DEFAULT_MODEL)
            frames, timestamps = extract_frames(video_path, sample_fps)
            embeddings = get_embeddings(frames, processor, clip_model)
            index = build_faiss_index(embeddings)

            results, result_timestamps, scores = search_frames(query, index, clip_model, processor, frames, timestamps, top_k=top_k)
            st.success(f"✅ Found {len(results)} matches.")

            # Display storyboard
            st.subheader("🖼️ Matched Frames")
            pil_images = [Image.fromarray(f) for f in results]
            st.image(pil_images, caption=[f"{t:.2f}s" for t in result_timestamps], use_container_width=True)

            storyboard = make_storyboard(pil_images)
            if storyboard:
                st.image(storyboard, caption="Storyboard of retrieved frames")

            # Get highlight clips
            st.subheader("🎬 Video Highlights")
            clips, merged_clip = get_clip_segments(video_path, result_timestamps, clip_duration)
            for idx, (clip_bytes, ts) in enumerate(zip(clips, result_timestamps), start=1):
                st.video(clip_bytes)
                st.caption(f"Clip #{idx} | Timestamp: {ts:.2f}s")

            if merged_clip:
                st.video(merged_clip)
                st.caption("Merged highlight reel")

            # Gemini AI Summary
            st.subheader("🧠 Gemini Video Summary")
            if st.button("Generate AI Summary"):
                with st.spinner("🤖 Analyzing with Gemini..."):
                    summary = get_gemini_summary(query, results, result_timestamps, merged_clip)
                    st.markdown(f"### Summary\n{summary}")

        finally:
            os.remove(video_path)


if __name__ == "__main__":
    main()
