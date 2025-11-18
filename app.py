# EchoVerse - Production ready single-file Streamlit app (Neumorphism UI)
# Option B - Neumorphism (soft, 3D, minimalist) UI redesign
# Features implemented:
# - High-end Neumorphism CSS (glass-like soft shadows, subtle 3D)
# - Premium layout: hero banner, feature cards, side-by-side panels
# - Reworked buttons with hover/active effects and micro-animations
# - Simple iconography using inline SVG + emojis
# - Elevated typography and spacing
# - Retains EmotionAdaptive voice, English only

import streamlit as st
from pathlib import Path
import os
import json
import tempfile
import time
import io
import base64
import subprocess
import shutil
from typing import List, Dict, Any, Tuple

# ML / TTS imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import pyttsx3

# ----------------------------
# Basic config
# ----------------------------
st.set_page_config(page_title="EchoVerse", page_icon="🎧", layout="wide")
BASE_DIR = Path(".")
LIB_DIR = BASE_DIR / "echoverse_library"
LIB_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_JSON = LIB_DIR / "projects_index.json"
if not PROJECTS_JSON.exists():
    PROJECTS_JSON.write_text("[]")

# Only English
LANG_CODE = {"English": "en"}

# ----------------------------
# Neumorphism CSS (soft UI)
# ----------------------------
st.markdown(
    r"""
<style>
:root{
  --bg1: #e8f4ff;
  --bg2: #fff5f8;
  --card: #fbfdff;
  --muted: #6b7280;
  --primary: #7c9cff;
  --primary-dark: #5b7df2;
  --accent: #9ee7c8;
  --glass: rgba(255,255,255,0.72);
  --soft-shadow: 12px 12px 28px rgba(160,170,190,0.25);
  --soft-highlight: -8px -8px 20px rgba(255,255,255,0.9);
  --btn-text: #ffffff;
}

/* Background */
html, body, .stApp, .main, .block-container {
  background: linear-gradient(180deg, var(--bg1), var(--bg2)) !important;
  background-attachment: fixed !important;
}

/* Container styles */
.navbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:18px 22px;
  margin-bottom:22px;
  border-radius:18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.55));
  box-shadow: var(--soft-shadow), var(--soft-highlight);
}
.brand { font-family: 'Helvetica Neue', Arial, sans-serif; font-weight:800; color: #0b2447; font-size: 24px; letter-spacing:0.6px; }
.subbrand { color:var(--muted); font-weight:600; margin-left:8px; font-size:13px }

/* Cards */
.card { background: linear-gradient(180deg, var(--card), #eef7ff); border-radius:16px; padding:20px; box-shadow: var(--soft-shadow), var(--soft-highlight); margin-bottom:20px; }
.hero { display:flex; gap:18px; align-items:center; padding:28px; border-radius:18px; }
.hero-left{ flex:1; }
.hero-right{ width:340px; }
h1.hero-title{ margin:0; font-size:30px; color:#07203a; font-weight:800; }
.punch{ margin-top:8px; color:#374151; font-size:15px }
.badge{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; background:linear-gradient(90deg,var(--accent), var(--primary)); color:#012; border-radius:999px; font-weight:700; box-shadow: 6px 6px 18px rgba(124,156,255,0.12); }

/* Buttons */
button.stButton>button, .st-audio .stButton>button {
  background: linear-gradient(180deg,var(--primary), var(--primary-dark)) !important;
  border:none !important;
  color: var(--btn-text) !important;
  padding:10px 16px !important;
  border-radius:12px !important;
  font-weight:700 !important;
  box-shadow: 6px 6px 18px rgba(108,132,255,0.12) !important;
  transition: transform .14s ease, box-shadow .14s ease !important;
}
button.stButton>button:hover, .st-audio .stButton>button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 12px 12px 28px rgba(108,132,255,0.18) !important;
}
button.stButton>button:active, .st-audio .stButton>button:active {
  transform: translateY(0) !important;
  box-shadow: 4px 4px 12px rgba(108,132,255,0.08) !important;
}

/* Download links */
.link-button{ display:inline-block; padding:8px 12px; border-radius:10px; background: linear-gradient(90deg,var(--primary), var(--primary-dark)); color:var(--btn-text); font-weight:700; text-decoration:none; }

.stTextArea textarea{ border-radius:12px !important; padding:14px !important; background: rgba(255,255,255,0.95) !important; }
.stFileUploader{ padding:6px; }

.small-muted{ color:var(--muted); font-size:13px }
.control-label{ font-weight:700; color:#07203a }
.option-card{ padding:12px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.55)); box-shadow: 8px 8px 20px rgba(160,170,190,0.06); }

.project-item{ padding:14px; border-radius:12px; margin-bottom:12px; display:flex; gap:12px; align-items:center; }
.project-thumb{ width:56px; height:56px; border-radius:12px; display:flex; align-items:center; justify-content:center; background: linear-gradient(180deg,#ffffff, #f2f7ff); box-shadow: 8px 8px 20px rgba(160,170,190,0.06); font-weight:800 }

@media(max-width:900px){ .hero{ flex-direction:column } .hero-right{ width:100% } }
@keyframes floaty{ 0%{ transform: translateY(0) } 50%{ transform: translateY(-4px) } 100%{ transform: translateY(0) } }
.pulse{ animation: floaty 4s ease-in-out infinite }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Navbar
# ----------------------------
nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    st.markdown(
        '<div class="navbar"><div><span class="brand">EchoVerse</span> <span class="subbrand">· Emotion-aware Audiobooks</span></div><div class="badge">✨ Expressive • Private • Local</div></div>',
        unsafe_allow_html=True,
    )
with nav_col2:
    btn_home, btn_create, btn_library = st.columns(3)
    with btn_home:
        if st.button("Home", key="nav_home"):
            st.query_params.update({"page": "Home"})
            st.rerun()
    with btn_create:
        if st.button("Create", key="nav_create"):
            st.query_params.update({"page": "Create"})
            st.rerun()
    with btn_library:
        if st.button("Library", key="nav_library"):
            st.query_params.update({"page": "Library"})
            st.rerun()

# Header
st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
st.markdown('<div class="card hero">', unsafe_allow_html=True)
left, right = st.columns([2, 1])
with left:
    st.markdown(
        '<div class="hero-left"><h1 class="hero-title">EchoVerse - An AI Audiobook Creation Tool</h1><div class="punch">Bring text to life with soft, human-like narrations that adapt to emotion — now in a calm, modern interface.</div><div style="height:12px"></div><div class="small-muted">Create audiobooks in English. Local-first storage. Emotion-adaptive narration for more expressive listening.</div></div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="hero-right" style="text-align:right"><div class="badge pulse">Neumorphism • Soft UI</div><div style="height:8px"></div><div class="small-muted">Tip: For best emotion control, use the offline engine.</div></div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Helper functions
# ----------------------------
def read_projects() -> List[Dict[str, Any]]:
    try:
        return json.loads(PROJECTS_JSON.read_text())
    except Exception:
        return []

def write_projects(projs: List[Dict[str, Any]]):
    PROJECTS_JSON.write_text(json.dumps(projs, indent=2))

def save_bytes_to_file(b: bytes, filename: str) -> str:
    path = LIB_DIR / filename
    with open(path, "wb") as f:
        f.write(b)
    return str(path)

def analyze_emotion(text: str) -> str:
    t = text.lower()
    interrogatives = ("who ", "what ", "when ", "where ", "why ", "how ")
    if "?" in t or t.strip().startswith(interrogatives):
        return "question"
    happy_words = ["happy","joy","excited","delighted","smile","wonderful","amazing","cheerful","thrilled","love"]
    for w in happy_words:
        if w in t:
            return "happy"
    if "!" in t:
        return "happy"
    sad_words = ["sad","sorrow","cry","unhappy","depressed","mourn","tears","lonely","tragic"]
    for w in sad_words:
        if w in t:
            return "sad"
    angry_words = ["angry","rage","hate","furious","enraged","shout","screamed","damn"]
    for w in angry_words:
        if w in t:
            return "angry"
    return "neutral"

def mp3_bytes_from_gtts(text: str, lang_code: str = "en") -> bytes:
    t = gTTS(text=text, lang=lang_code)
    fp = io.BytesIO()
    t.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def mp3_bytes_from_pyttsx3(text: str, lang: str = "English", emotion: str = "neutral") -> Tuple[bytes, str]:
    tmp_dir = Path(tempfile.gettempdir())
    ts = int(time.time() * 1000)
    tmp_wav = tmp_dir / f"echoverse_{ts}.wav"
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    volume = engine.getProperty("volume")
    voices = engine.getProperty("voices")
    preferred_keyword = None
    if emotion == "happy":
        rate = int(rate * 1.15); volume = min(1.0, volume + 0.1); preferred_keyword = "zira"
    elif emotion == "sad":
        rate = int(rate * 0.75); volume = max(0.55, volume - 0.22); preferred_keyword = "zira"
    elif emotion == "angry":
        rate = int(rate * 1.05); volume = 1.0; preferred_keyword = "david"
    elif emotion == "question":
        rate = int(rate * 1.08); volume = min(1.0, volume + 0.05); preferred_keyword = "zira"
    selected_voice = None
    if preferred_keyword:
        for v in voices:
            if preferred_keyword in (v.name or "").lower() or preferred_keyword in (v.id or "").lower():
                selected_voice = v.id
                break
    if selected_voice:
        try:
            engine.setProperty("voice", selected_voice)
        except Exception:
            pass
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    try:
        engine.save_to_file(text, str(tmp_wav))
        engine.runAndWait()
    except Exception as e:
        raise RuntimeError(f"pyttsx3 failed: {e}")
    if not tmp_wav.exists():
        raise RuntimeError("pyttsx3 did not produce an output file (wav).")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        tmp_conv = tmp_dir / f"echoverse_conv_{ts}.mp3"
        cmd = [ffmpeg_path, "-y", "-loglevel", "error", "-i", str(tmp_wav), str(tmp_conv)]
        try:
            subprocess.run(cmd, check=True)
            if tmp_conv.exists():
                data = tmp_conv.read_bytes()
                tmp_wav.unlink(missing_ok=True)
                tmp_conv.unlink(missing_ok=True)
                return data, "mp3"
        except subprocess.CalledProcessError:
            pass
    data = tmp_wav.read_bytes()
    tmp_wav.unlink(missing_ok=True)
    return data, "wav"

@st.cache_resource(ttl=60 * 60 * 24)
def load_granite(model_name: str = "ibm-granite/granite-3.2-2b-instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    return tokenizer, model, device

def granite_generate(tokenizer, model, device, messages: List[Dict[str, str]], max_new_tokens: int = 200) -> str:
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, temperature=0.7)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()

# ----------------------------
# Navigation & Username
# ----------------------------
qp = st.query_params
page = qp.get("page", ["Home"])[0] if isinstance(qp.get("page"), list) else qp.get("page", "Home")

if "username" not in st.session_state:
    st.session_state["username"] = ""
if not st.session_state["username"]:
    name_input = st.text_input("Enter your display name to begin", key="name_input")
    if name_input:
        st.session_state["username"] = name_input.strip()
        st.rerun()
else:
    st.markdown(f"### Hello, **{st.session_state['username']}** 👋")

# ----------------------------
# Pages
# ----------------------------
if page == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Why EchoVerse — Soft, expressive narration")
    st.write("• Emotion-adaptive narration — voice automatically softens, brightens, or firms up based on content.")
    st.write("• Local-first: audio and metadata saved on your machine for privacy.")
    st.write("• Neumorphism UI: clean, soft, and focused for long writing sessions.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Create":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Create Audiobook")
    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader("Upload .txt file (optional)", type=["txt"])
        if uploaded:
            original_text = uploaded.read().decode("utf-8")
        else:
            original_text = st.text_area("Or paste your original text here", height=340)
        if not original_text or original_text.strip() == "":
            st.info("Please provide text to continue.")

    with right:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Options</div>', unsafe_allow_html=True)
        tone = st.selectbox("Rewrite tone (display only)", ["Neutral", "Suspenseful", "Inspiring"])
        tts_engine = st.selectbox("TTS engine", ["gTTS (online, mp3)", "pyttsx3 (offline)"])
        lang = st.selectbox("Audio language", ["English"])
        emotion_adaptive = st.checkbox("EmotionAdaptive Voice (enabled)", value=True)
        gen_tokens = st.slider("Granite generation tokens", 50, 800, 250, step=50)
        custom_fname = st.text_input("Custom audiobook name (without extension)", value=f"echoverse_{st.session_state.get('username','user')}")
        st.markdown("<div class='small-muted'>Tip: EmotionAdaptive uses pyttsx3 for more expressive local control.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if original_text and original_text.strip():
        st.markdown("</div><div class='card'>", unsafe_allow_html=True)
        st.subheader("Generate audio from ORIGINAL text")
        if st.button("Generate Audio", key="generate_audio"):
            with st.spinner("Composing expressive narration..."):
                text_for_tts = original_text
                emotion = analyze_emotion(text_for_tts)
                if emotion_adaptive and tts_engine.startswith("gTTS"):
                    st.info("Switching to pyttsx3 for emotion-adaptive voice control.")
                    tts_engine = "pyttsx3 (offline)"
                audio_bytes = None
                audio_ext = "mp3"
                try:
                    if tts_engine.startswith("gTTS"):
                        audio_bytes = mp3_bytes_from_gtts(text_for_tts, LANG_CODE.get(lang, "en"))
                        audio_ext = "mp3"
                    else:
                        audio_bytes, audio_ext = mp3_bytes_from_pyttsx3(text_for_tts, lang=lang, emotion=emotion)
                except Exception as e:
                    st.error(f"TTS failed: {e}")
                if audio_bytes:
                    fname = f"{custom_fname}.{audio_ext}"
                    saved_path = save_bytes_to_file(audio_bytes, fname)
                    st.success(f"Audio generated and saved as {fname}")
                    mime = "audio/mpeg" if audio_ext == "mp3" else "audio/wav"
                    st.audio(audio_bytes, format=mime)
                    unique_key = f"dl_gen_{int(time.time() * 1000)}"
                    if audio_ext == "mp3":
                        st.download_button("Download MP3", data=audio_bytes, file_name=fname, mime="audio/mpeg", key=unique_key)
                    else:
                        st.download_button("Download audio (wav)", data=audio_bytes, file_name=fname, mime="audio/wav", key=unique_key)
                    projs = read_projects()
                    projs.insert(
                        0,
                        {
                            "id": int(time.time()),
                            "name": fname,
                            "owner": st.session_state.get("username", ""),
                            "tone": tone,
                            "language": lang,
                            "emotion_detected": emotion,
                            "filename": fname,
                            "path": saved_path,
                            "original_preview": original_text[:250],
                            "rewritten_preview": st.session_state.get("last_rewrite", "")[:250],
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )
                    write_projects(projs)
                else:
                    st.error("Audio generation failed.")

elif page == "Library":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Library — Saved Projects")
    projects = read_projects()
    if not projects:
        st.info("No saved audiobooks yet. Create one on the Create page.")
    else:
        for idx, p in enumerate(projects):
            with st.expander(f"{p.get('name')} — {p.get('created_at')}"):
                st.markdown('<div class="project-item">', unsafe_allow_html=True)
                st.markdown(f'<div class="project-thumb">🎧</div>', unsafe_allow_html=True)
                st.write(f"**{p.get('name')}**  — Owner: {p.get('owner')} | Tone: {p.get('tone')} | Language: {p.get('language')} | Emotion: {p.get('emotion_detected','-')}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.write("Original preview:")
                st.write(p.get("original_preview", ""))
                st.write("Rewritten preview:")
                st.write(p.get("rewritten_preview", ""))
                path = p.get("path")
                if path and Path(path).exists():
                    audio_bytes = Path(path).read_bytes()
                    ext = Path(path).suffix.lower().lstrip(".")
                    mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
                    st.audio(audio_bytes, format=mime)
                    dl_key = f"dl_lib_{p.get('id')}_{idx}_{int(time.time()*1000)}"
                    if ext == "mp3":
                        st.download_button("Download MP3", data=audio_bytes, file_name=p.get("name", "audio.mp3"), mime="audio/mpeg", key=dl_key)
                    else:
                        st.download_button("Download audio (wav)", data=audio_bytes, file_name=p.get("name", "audio.wav"), mime="audio/wav", key=dl_key)
                else:
                    st.warning("Audio file missing from disk.")
                if st.button("Delete", key=f"del_{p.get('id')}_{idx}"):
                    try:
                        if path and Path(path).exists():
                            Path(path).unlink()
                        new_list = [x for x in projects if x.get("id") != p.get("id")]
                        write_projects(new_list)
                        st.success("Deleted project.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer / debug
st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
if st.checkbox("Show debug info (model availability)"):
    try:
        tk, md, dev = load_granite()
        st.write("Granite loaded. Device:", dev)
    except Exception as e:
        st.write("Granite not loaded:", e)
