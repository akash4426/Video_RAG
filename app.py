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
class DevNull:
    def write(self, _): pass
    def flush(self): pass
sys.stderr = DevNull()

# ---- Optional: fix MoviePy IM path issue ----
import moviepy.config_defaults as cfg
cfg.IMAGEMAGICK_BINARY = None

# =========================================================
# 1. IMPORTS
# =========================================================
import streamlit as st
import cv2
import torch
import numpy as np
import faiss
import time
import tempfile
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =========================================================
# 2. ERROR HANDLER
# =========================================================
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

# =========================================================
# 3. DEVICE SETUP
# =========================================================
@handle_errors
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
st.sidebar.info(f"⚙️ Using device: **{str(device).upper()}**")

# =========================================================
# 4. LOAD MODELS
# =========================================================
@st.cache_resource(ttl=3600)
@handle_errors
def load_clip_model():
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_fast=True
    )
    model.eval()
    return model, processor

clip_model, processor = load_clip_model()

@st.cache_resource(ttl=3600)
def load_asr_model():
    # small or tiny — tiny runs faster on CPU cloud
    return WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)

@st.cache_resource(ttl=3600)
def load_text_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================================================
# 5. FRAME EXTRACTION
# =========================================================
@handle_errors
def extract_frames(video_path, sample_fps=1):
    frames, ts = [], []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # fallback FPS
            
        step = max(1, int(round(fps / sample_fps)))
        count = 0
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                ts.append(count / fps)
            count += 1
            
        if not frames:
            raise ValueError("No frames extracted from video")
            
        return frames, ts
    
    finally:
        if cap is not None:
            cap.release()

# =========================================================
# 6. CLIP EMBEDDINGS
# =========================================================
def get_embeddings(frames, batch_size=8):
    all_feats = []
    prog = st.progress(0, text="🔄 Embedding frames...")
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        prog.progress(min((i+batch_size)/len(frames),1.0))
    prog.empty()
    return np.vstack(all_feats).astype("float32")

# =========================================================
# 7. TEMPORAL AGGREGATION
# =========================================================
def aggregate_to_clips(frame_embs, ts, window_sec):
    if len(ts)==0:
        return np.zeros((0,frame_embs.shape[1]),dtype="float32"),[],[]
    clips, bounds, centers = [], [], []
    i=0
    while i < len(ts):
        start_t = ts[i]; end_t = start_t + window_sec
        j=i
        while j < len(ts) and ts[j]<=end_t: j+=1
        group = frame_embs[i:j]
        clips.append(group.mean(axis=0))
        bounds.append((start_t, ts[j-1] if j-1<len(ts) else end_t))
        centers.append(0.5*(start_t+(ts[j-1] if j-1<len(ts) else end_t)))
        i=j
    clips = np.vstack(clips).astype("float32")
    return clips,bounds,centers

# =========================================================
# 8. WHISPER + TEXT EMBEDDINGS
# =========================================================
def transcribe(video_path):
    model = load_asr_model()
    segs_it, _ = model.transcribe(video_path, beam_size=1, vad_filter=True)
    segs=[]
    for s in segs_it:
        segs.append({"start":float(s.start),"end":float(s.end),"text":s.text.strip()})
    return segs

def embed_segments(segs):
    if not segs: return np.zeros((0,384),dtype="float32")
    model = load_text_embedder()
    texts = [s["text"] for s in segs]
    embs = model.encode(texts, normalize_embeddings=True)
    return embs.astype("float32")

def nearest_seg(segs, t):
    best,bestd=-1,1e9
    for i,s in enumerate(segs):
        if s["start"]<=t<=s["end"]: return i
        mid=(s["start"]+s["end"])/2; d=abs(mid-t)
        if d<bestd:bestd=d;best=i
    return best

# =========================================================
# 9. FUSE VISUAL + TEXT
# =========================================================
def fuse_modalities(vembs, bounds, segs, tembs, wv, wt):
    fused,centers=[],[]
    for i,(s,e) in enumerate(bounds):
        tc=0.5*(s+e); centers.append(tc)
        if not segs or tembs.shape[0]==0:
            f=vembs[i]
        else:
            j=nearest_seg(segs,tc)
            v,t=vembs[i],tembs[j]
            if t.shape[0]!=v.shape[0]:
                if t.shape[0]<v.shape[0]:
                    t=np.pad(t,(0,v.shape[0]-t.shape[0]))
                else:t=t[:v.shape[0]]
            f=(wv*v+wt*t)/(wv+wt)
        f=f/np.linalg.norm(f)
        fused.append(f.astype("float32"))
    return np.vstack(fused),centers

# =========================================================
# 10. FAISS INDEX
# =========================================================
def build_index(embs):
    d=embs.shape[1]; faiss.normalize_L2(embs)
    idx=faiss.IndexFlatIP(d); idx.add(embs); return idx

# =========================================================
# 11. QUERY EXPANSION (optional)
# =========================================================
def expand_query(q):
    try:
        key=st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=key)
        model=genai.GenerativeModel("gemini-2.5-flash")
        p=f"Expand the following query into 3 similar variants for video retrieval:\n{q}"
        r=model.generate_content(p)
        lines=[l.strip("- ").strip() for l in r.text.split("\n") if l.strip()]
        return [q]+lines[:3]
    except Exception:
        return [q]

def encode_clip_text(q):
    ti=processor(text=[q],return_tensors="pt",padding=True).to(device)
    with torch.no_grad():
        t=clip_model.get_text_features(**ti)
        t=t/t.norm(p=2,dim=-1,keepdim=True)
    return t.cpu().numpy().astype("float32")

# =========================================================
# 12. GEMINI SUMMARY
# =========================================================
def get_summary(q,frames,tss,texts):
    try:
        key=st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=key)
        model=genai.GenerativeModel("gemini-2.5-flash")
        imgs=[Image.fromarray(f) for f in frames]
        txt="\n".join([f"[{t:.2f}s] {tx}" for t,tx in zip(tss,texts) if tx])
        prompt=f"Query: {q}\nUsing visuals and transcript:\n{txt}\nSummarize scene context and key actions."
        parts=[prompt]+imgs
        r=model.generate_content(parts)
        return r.text
    except Exception as e:
        return f"Gemini error: {e}"

# =========================================================
# 13. CLIP EXTRACTION UTILITY
# =========================================================
@handle_errors
def get_clip_segments(video_path, tss, win):
    clips = []
    video = None
    try:
        video = VideoFileClip(video_path)
        for t in tss:
            start_time = max(0, t - win/2)
            end_time = min(video.duration, t + win/2)
            subclip = video.subclip(start_time, end_time)
            temp_path = tempfile.mktemp(suffix='.mp4')
            subclip.write_videofile(
                temp_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                logger=None,
                verbose=False
            )
            with open(temp_path, "rb") as f:
                clip_bytes = f.read()
            clips.append({
                "bytes": clip_bytes,
                "start": start_time,
                "end": end_time
            })
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return clips
    finally:
        if video is not None:
            try: video.close()
            except: pass

# =========================================================
# 14. STREAMLIT UI
# =========================================================
def main():
    st.title("🎥 Multi-Modal Temporal Video RAG")
    st.write("Search inside videos using **CLIP + Whisper + Gemini**")

    up = st.file_uploader("📁 Upload a video", type=["mp4","mov","avi"])
    q  = st.text_input("🔍 Enter query", "person talking on phone")
    fps= st.slider("🎞 Sampling FPS",0.5,5.0,1.0)
    win= st.slider("🎬 Clip window (sec)",1.0,5.0,2.0)
    wv = st.slider("Weight visual",0.0,1.0,0.6,0.05)
    wt = st.slider("Weight transcript",0.0,1.0,0.4,0.05)
    exp= st.checkbox("Use query expansion",True)

    if up and q:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
            tmp.write(up.read()); path=tmp.name
        try:
            with st.spinner("Processing video..."):
                frames,ts=extract_frames(path,fps)
                frame_emb=get_embeddings(frames)
                clip_vis,clip_bounds,centers=aggregate_to_clips(frame_emb,ts,win)
                segs=transcribe(path)
                txt_emb=embed_segments(segs)
                fused,clip_centers=fuse_modalities(clip_vis,clip_bounds,segs,txt_emb,wv,wt)
                idx=build_index(fused)

                queries=expand_query(q) if exp else [q]
                hits={}
                for qq in queries:
                    tvec=encode_clip_text(qq)
                    D,I=idx.search(tvec,3)
                    for s,i in zip(D[0],I[0]): hits[i]=max(hits.get(i,0),float(s))
                best=sorted(hits.items(),key=lambda x:x[1],reverse=True)[:3]
                tss=[clip_centers[i] for i,_ in best]; scs=[s for _,s in best]

                clips=get_clip_segments(path,tss,win)
                st.success(f"Found {len(clips)} relevant clips.")
                for i,(cl,tsc,sc) in enumerate(zip(clips,tss,scs)):
                    st.subheader(f"Match {i+1} | {tsc:.2f}s | Score {sc:.3f}")
                    st.video(cl["bytes"])
                    st.caption(f"Clip timeframe {cl['start']:.2f}-{cl['end']:.2f}s")
                    segtxt=segs[nearest_seg(segs,tsc)]["text"] if segs else ""
                    if segtxt: st.caption(f"🗣 {segtxt}")

                with st.expander("🧠 Gemini Summary"):
                    if st.button("Generate Summary"):
                        txts=[segs[nearest_seg(segs,t)]["text"] if segs else "" for t in tss]
                        st.write(get_summary(q,[frames[0] for _ in tss],tss,txts))
        finally:
            if os.path.exists(path): os.unlink(path)

if __name__=="__main__":
    main()
