# app.py
"""
EchoVerse - Single-file Streamlit app (no pydub).
Features:
- Tone-adaptive rewriting with local IBM Granite LLM (HuggingFace model)
- TTS via gTTS (recommended) or pyttsx3 (offline)
- Saves projects (audio + metadata) to local library
- Interactive Q&A about original text using Granite
- Visily-like UI styling (gradient navbar, cards)
- Voice preference and custom audiobook naming added
"""

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

LANG_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Kannada": "kn",
    "Tamil": "ta",
    "Malayalam": "ml",
}

# ----------------------------
# UI CSS (Visily-inspired)
# ----------------------------
st.markdown(
    """
    <style>
    body { background-color: #f6f8fb; }
    .navbar {
        background: linear-gradient(90deg,#6a11cb,#2575fc);
        color: white;
        padding: 14px 22px;
        border-radius: 8px;
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:18px;
    }
    .brand { font-size:22px; font-weight:700; color:white; }
    .nav-btns { display:flex; gap:12px; }
    .card {
        background:white; padding:20px; border-radius:12px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.06); margin-bottom:18px;
    }
    .hero { background: linear-gradient(135deg,#eef2ff,#e6f0ff); padding:34px; border-radius:12px; text-align:center; margin-bottom:18px; }
    div.stButton > button { background:linear-gradient(90deg,#6a11cb,#2575fc); color:white; border-radius:8px; padding:8px 14px; border:none; }
    .small-muted { color: #6b7280; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Streamlit-native Navbar
# ----------------------------
nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    st.markdown('<div class="brand">EchoVerse</div>', unsafe_allow_html=True)
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

# ----------------------------
# AI Tool Title Bar
# ----------------------------
st.markdown('<h2 style="margin-top:0;color:#6a11cb">🎧 EchoVerse AI Audiobook Tool</h2>', unsafe_allow_html=True)

# ----------------------------
# Helper functions
# ----------------------------
def read_projects() -> List[Dict[str,Any]]:
    try:
        return json.loads(PROJECTS_JSON.read_text())
    except Exception:
        return []

def write_projects(projs: List[Dict[str,Any]]):
    PROJECTS_JSON.write_text(json.dumps(projs, indent=2))

def save_bytes_to_file(b: bytes, filename: str) -> str:
    path = LIB_DIR / filename
    with open(path, "wb") as f:
        f.write(b)
    return str(path)

def make_mp3_download_link(mp3_bytes: bytes, filename: str, label: str = "Download MP3"):
    b64 = base64.b64encode(mp3_bytes).decode()
    href = f'<a href="data:audio/mpeg;base64,{b64}" download="{filename}">{label}</a>'
    return href

# ----------------------------
# TTS functions
# ----------------------------
def mp3_bytes_from_gtts(text: str, lang_code: str = "en") -> bytes:
    t = gTTS(text=text, lang=lang_code)
    fp = io.BytesIO()
    t.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def mp3_bytes_from_pyttsx3(text: str, preferred_gender: str = "Female") -> Tuple[bytes, str]:
    """
    Create MP3 (or WAV fallback) using pyttsx3 with reliable male/female voice selection on Windows.
    """
    tmp_dir = Path(tempfile.gettempdir())
    ts = int(time.time()*1000)
    tmp_mp3 = tmp_dir / f"echoverse_{ts}.mp3"
    tmp_wav = tmp_dir / f"echoverse_{ts}.wav"

    engine = pyttsx3.init()
    
    # --- Voice selection ---
    voices = engine.getProperty("voices")
    selected_voice = None

    # Predefined SAPI5 IDs for Windows (adjust if your system differs)
    WINDOWS_VOICES = {
        "female": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens\\MSTTS_V110_enUS_ZiraM",
        "male":   "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens\\MSTTS_V110_enUS_DavidM"
    }

    # Try to use the preferred gender voice
    if preferred_gender.lower() in WINDOWS_VOICES:
        for v in voices:
            if v.id == WINDOWS_VOICES[preferred_gender.lower()]:
                selected_voice = v.id
                break

    # Fallback: pick first voice that contains gender keyword
    if not selected_voice:
        for v in voices:
            name = (v.name or "").lower()
            if preferred_gender.lower() == "female" and "zira" in name:
                selected_voice = v.id
                break
            if preferred_gender.lower() == "male" and "david" in name:
                selected_voice = v.id
                break

    # Set the voice if found
    if selected_voice:
        engine.setProperty("voice", selected_voice)
    
    # --- Generate audio ---
    try:
        # Attempt MP3 first
        engine.save_to_file(text, str(tmp_mp3))
        engine.runAndWait()
        if tmp_mp3.exists() and tmp_mp3.stat().st_size > 0:
            data = tmp_mp3.read_bytes()
            tmp_mp3.unlink(missing_ok=True)
            return data, "mp3"
    except Exception:
        pass

    # Fallback WAV
    try:
        engine.save_to_file(text, str(tmp_wav))
        engine.runAndWait()
    except Exception as e:
        raise RuntimeError(f"pyttsx3 failed: {e}")

    if not tmp_wav.exists():
        raise RuntimeError("pyttsx3 did not produce any output (wav).")

    # Convert WAV → MP3 if ffmpeg available
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

    # Return WAV as last resort
    data = tmp_wav.read_bytes()
    tmp_wav.unlink(missing_ok=True)
    return data, "wav"

    tmp_dir = Path(tempfile.gettempdir())
    ts = int(time.time()*1000)
    tmp_mp3 = tmp_dir / f"echoverse_{ts}.mp3"
    tmp_wav = tmp_dir / f"echoverse_{ts}.wav"

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    selected = None
    for v in voices:
        name = (v.name or "").lower()
        if preferred_gender.lower() == "female" and "female" in name:
            selected = v.id
            break
        if preferred_gender.lower() == "male" and "male" in name:
            selected = v.id
            break
    if selected:
        engine.setProperty("voice", selected)

    try:
        engine.save_to_file(text, str(tmp_mp3))
        engine.runAndWait()
        if tmp_mp3.exists() and tmp_mp3.stat().st_size > 0:
            data = tmp_mp3.read_bytes()
            tmp_mp3.unlink(missing_ok=True)
            return data, "mp3"
    except Exception:
        pass

    try:
        if tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
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

# ----------------------------
# Granite LLM
# ----------------------------
@st.cache_resource(ttl=60*60*24)
def load_granite(model_name: str = "ibm-granite/granite-3.2-2b-instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    model.to(device)
    return tokenizer, model, device

def granite_generate(tokenizer, model, device, messages: List[Dict[str,str]], max_new_tokens: int = 200) -> str:
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, temperature=0.7)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()

# ----------------------------
# Navigation & Username
# ----------------------------
qp = st.query_params
_page_val = qp.get("page", None)
if isinstance(_page_val, list):
    page = _page_val[0] if len(_page_val) > 0 else "Home"
elif isinstance(_page_val, str):
    page = _page_val
else:
    page = "Home"

if "username" not in st.session_state:
    st.session_state["username"] = ""

if not st.session_state["username"]:
    name_input = st.text_input("Enter your name to begin", key="name_input")
    if name_input:
        st.session_state["username"] = name_input.strip()
        st.rerun()
else:
    st.markdown(f"### Hello, **{st.session_state['username']}** 👋")

# ----------------------------
# PAGES
# ----------------------------

# HOME
if page == "Home":
    st.markdown('<div class="hero"><h1 style="margin:0">EchoVerse</h1><p class="small-muted">Transform text into expressive audiobooks — tone-aware rewriting, multi-language TTS, and a local library.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Quick Overview")
    st.write("""
    - Rewrites text into **Neutral**, **Suspenseful**, or **Inspiring** tones using a local Granite LLM (display-only).  
    - Creates downloadable audio from the **original** text (optionally translated) in English, Hindi, Telugu, Kannada, Tamil, Malayalam.  
    - Library to save projects (metadata + audio).  
    - Interactive mode: Ask questions about the original text and Granite answers on the fly.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# CREATE
elif page == "Create":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Create Audiobook")
    left, right = st.columns([2,1])
    with left:
        uploaded = st.file_uploader("Upload .txt file (optional)", type=["txt"])
        if uploaded:
            original_text = uploaded.read().decode("utf-8")
        else:
            original_text = st.text_area("Or paste your original text here", height=300)
        if not original_text or original_text.strip() == "":
            st.info("Please provide text to continue.")

    with right:
        st.markdown("### Options")
        tone = st.selectbox("Rewrite tone (display only)", ["Neutral", "Suspenseful", "Inspiring"])
        tts_engine = st.selectbox("TTS engine", ["gTTS (online, mp3)", "pyttsx3 (offline, may produce wav)"])
        lang = st.selectbox("Audio language", list(LANG_CODE.keys()))
        gender = st.selectbox("Voice preference", ["Female", "Male"])
        gen_tokens = st.slider("Granite generation tokens", 50, 800, 250, step=50)
        custom_fname = st.text_input("Custom audiobook name (without extension)", value=f"echoverse_{st.session_state.get('username','user')}")
        st.markdown('<div class="small-muted">Note: Rewritten text is display-only and not used for audio generation.</div>', unsafe_allow_html=True)

    if original_text and original_text.strip():
        # … side-by-side rewritten text and TTS generation code …
        st.markdown("</div><div class='card'>", unsafe_allow_html=True)
        st.subheader("Generate audio from ORIGINAL text (translated if needed)")

        if st.button("Generate Audio"):
            with st.spinner("Preparing audio..."):
                text_for_tts = original_text
                if lang != "English":
                    try:
                        tokenizer, model, device = load_granite()
                        messages = [{"role":"system","content":"Translate preserving exact meaning and register."},
                                    {"role":"user","content":f"Translate the following text to {lang}:\n\n{original_text}"}]
                        translated = granite_generate(tokenizer, model, device, messages, max_new_tokens=1000)
                        if translated.strip():
                            text_for_tts = translated
                    except Exception as e:
                        st.warning(f"Granite translation failed: {e} — using original text.")

                audio_bytes = None
                audio_ext = "mp3"
                try:
                    if tts_engine.startswith("gTTS"):
                        audio_bytes = mp3_bytes_from_gtts(text_for_tts, LANG_CODE.get(lang,"en"))
                    else:
                        audio_bytes, audio_ext = mp3_bytes_from_pyttsx3(text_for_tts, preferred_gender=gender)
                except Exception as e:
                    st.error(f"TTS failed: {e}")

                if audio_bytes:
                    fname = f"{custom_fname}.{audio_ext}"
                    saved_path = save_bytes_to_file(audio_bytes, fname)
                    st.success(f"Audio generated and saved as {fname}")
                    st.audio(audio_bytes, format="audio/mpeg" if audio_ext=="mp3" else "audio/wav")
                    if audio_ext=="mp3":
                        st.markdown(make_mp3_download_link(audio_bytes, fname), unsafe_allow_html=True)
                    else:
                        st.download_button("Download audio (wav)", data=audio_bytes, file_name=fname, mime="audio/wav")
                    # Save metadata
                    projs = read_projects()
                    projs.insert(0, {
                        "id": int(time.time()),
                        "name": fname,
                        "owner": st.session_state.get("username",""),
                        "tone": tone,
                        "language": lang,
                        "gender": gender,
                        "filename": fname,
                        "path": saved_path,
                        "original_preview": original_text[:250],
                        "rewritten_preview": st.session_state.get("last_rewrite","")[:250],
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    write_projects(projs)
                else:
                    st.error("Audio generation failed.")

# LIBRARY
elif page == "Library":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Library — Saved Projects")
    projects = read_projects()
    if not projects:
        st.info("No saved audiobooks yet. Create one on the Create page.")
    else:
        for p in projects:
            with st.expander(f"{p.get('name')} — {p.get('created_at')}"):
                st.write(f"Owner: {p.get('owner')} | Tone: {p.get('tone')} | Language: {p.get('language')}")
                st.write("Original preview:")
                st.write(p.get("original_preview",""))
                st.write("Rewritten preview:")
                st.write(p.get("rewritten_preview",""))
                path = p.get("path")
                if path and Path(path).exists():
                    audio_bytes = Path(path).read_bytes()
                    ext = Path(path).suffix.lower().lstrip(".")
                    mime = "audio/mpeg" if ext=="mp3" else "audio/wav"
                    st.audio(audio_bytes, format=mime)
                    if ext=="mp3":
                        st.markdown(make_mp3_download_link(audio_bytes, p.get("name","audio.mp3")), unsafe_allow_html=True)
                    else:
                        st.download_button("Download audio (wav)", data=audio_bytes, file_name=p.get("name","audio.wav"), mime="audio/wav")
                else:
                    st.warning("Audio file missing from disk.")
                if st.button("Delete", key=f"del_{p.get('id')}"):
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
st.markdown("<br><br>", unsafe_allow_html=True)
if st.checkbox("Show debug info (model availability)"):
    try:
        tk, md, dev = load_granite()
        st.write("Granite loaded. Device:", dev)
    except Exception as e:
        st.write("Granite not loaded:", e)

