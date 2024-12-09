import streamlit as st
import os
import tempfile
import subprocess
import openai
from transformers import pipeline

def set_openai_api_key(api_key):
    openai.api_key = api_key

@st.cache_resource
def load_translation_model(source_lang, target_lang):
    model_id = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    return pipeline("translation", model=model_id, tokenizer=model_id)

def extract_audio_to_mp3(video_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_temp:
            video_temp.write(video_file.getbuffer())
            video_temp_path = video_temp.name

        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_output_path = audio_temp.name

        command = [
            "ffmpeg", "-i", video_temp_path, "-q:a", "0", "-map", "a", audio_output_path, "-y"
        ]
        subprocess.run(command, check=True)
        os.remove(video_temp_path)
        return audio_output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def generate_subtitles(audio_path, api_key):
    set_openai_api_key(api_key)
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe(
            file=f,
            model="whisper-1",
            response_format="srt"
        )
    return transcript

def translate_subtitles(subtitles, source_lang, target_lang):
    if source_lang == target_lang:
        return subtitles

    try:
        translator = load_translation_model(source_lang, target_lang)
        translated_lines = []
        for line in subtitles.split("\n"):
            if line.strip() and not line.strip().isdigit() and "-->" not in line:
                translation = translator(line, max_length=512, truncation=True)
                translated_lines.append(translation[0]["translation_text"])
            else:
                translated_lines.append(line)
        return "\n".join(translated_lines)
    except Exception as e:
        raise RuntimeError(f"Error during translation: {e}")

language_code_map = {
    "English": "en",
    "Russian": "ru",
    "Ukrainian": "uk",
    "Polish": "pl"
}

st.sidebar.header("APP control panel")
api_key = st.sidebar.text_input("", type="password", placeholder="Enter your API key")
confirm_api_key = st.sidebar.button("OK")

if "api_key_confirmed" not in st.session_state:
    st.session_state["api_key_confirmed"] = False

if confirm_api_key:
    if not api_key:
        st.sidebar.error("Please enter your API key to proceed.")
    else:
        st.session_state["api_key_confirmed"] = True
        st.toast("API key confirmed!", icon="🎉")

if not st.session_state["api_key_confirmed"]:
    st.stop()

source_language = st.sidebar.selectbox(
    "Select the original language of the video",
    options=["English", "Russian", "Ukrainian", "Polish"],
    index=0
)
source_language_code = language_code_map[source_language]

st.sidebar.write("Load your video file (Limit 100MB per file • MP4, MKV, AVI, MPEG4)")
uploaded_file = st.sidebar.file_uploader("", type=["mp4", "mkv", "avi", "mpeg4"])

target_language = st.sidebar.selectbox(
    "Select the target language for subtitles",
    options=["English", "Russian", "Ukrainian", "Polish"],
    index=0
) 
target_language_code = language_code_map[target_language]

create_subtitles_button = st.sidebar.button("Create Subtitles")

if "srt_content" not in st.session_state:
    st.session_state["srt_content"] = ""

st.markdown(
    """
    <h1 style='text-align: center; color: orange;'>Subtitler</h1>
    <p style='text-align: center; color: blue; font-size:18px;'>Built with passion By Zoe LAB 😘 (ver. 07).</p>
    """,
    unsafe_allow_html=True
)

if uploaded_file:
    st.video(uploaded_file)

if uploaded_file and create_subtitles_button:
    progress_bar = st.progress(0)
    try:
        progress_bar.progress(10)
        audio_path = extract_audio_to_mp3(uploaded_file)
        progress_bar.progress(30)

        subtitles = generate_subtitles(audio_path, api_key)
        progress_bar.progress(60)
       
        translated_subtitles = translate_subtitles(subtitles, source_language_code, target_language_code)
        progress_bar.progress(90)
        
        st.session_state["srt_content"] = translated_subtitles

        progress_bar.progress(100)
        st.toast("Subtitles successfully generated and translated!", icon="🎉")
        
        os.remove(audio_path)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.session_state["srt_content"]:
    edited_srt_content = st.text_area(
        "Edit your subtitles below:",
        value=st.session_state["srt_content"],
        height=300,
        key="subtitle_editor"
    )

    if edited_srt_content != st.session_state["srt_content"]:
        st.session_state["srt_content"] = edited_srt_content
    
    st.download_button(
        label="Download Final SRT",
        data=st.session_state["srt_content"],
        file_name="final_subtitles.srt",
        mime="text/srt"
    )