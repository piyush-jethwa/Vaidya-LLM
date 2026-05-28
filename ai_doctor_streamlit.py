import os
import tempfile
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Vaidya Ai - Healthcare assistant", layout="wide")

from brain_of_the_doctor import (
    encode_image,
    analyze_image_with_query,
    generate_prescription,
    analyze_text_query
)
try:
    from voice_of_the_patient import transcribe_with_groq
except ImportError as e:
    st.error(f"Error importing voice module: {e}")
    # Create a fallback function
    def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
        st.error("Voice transcription not available due to import error")
        return None

from gtts import gTTS
import base64
import io

@st.cache_data
def generate_audio_from_text(text, lang):
    """Generates audio from text using gTTS and caches the result."""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes_io = io.BytesIO()
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io.getvalue()
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None

# Get API key from environment variables (for Streamlit deployment)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Debug: Show API key status (without revealing the key)
if GROQ_API_KEY:
    st.info(f"🔑 API Key Status: Found (Length: {len(GROQ_API_KEY)} characters)")
    if GROQ_API_KEY.startswith("gsk_"):
        st.success("✅ API Key format looks correct (starts with 'gsk_')")
    else:
        st.warning("⚠️ API Key format may be incorrect (should start with 'gsk_')")
else:
    st.error("❌ No API key found! Please set GROQ_API_KEY environment variable in Streamlit deployment settings.")

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

TRANSLATIONS = {
    "English": {
        "title": "🩺 Vaidya Ai - Healthcare assistant",
        "subtitle": "Professional medical diagnosis powered by AI",
        "input": "Input",
        "voice_tab": "🎤 Voice Input",
        "text_tab": "✍️ Text Input",
        "describe_symptoms": "Describe your symptoms",
        "earlier_symptoms": "Earlier symptoms / what problem are you facing?",
        "days_suffering": "Days suffering",
        "days_help": "From how many days are you suffering?",
        "upload_image": "Upload Medical Image (Optional)",
        "doctor_panel": "Your Doctor",
        "get_diagnosis": "🔍 Get Diagnosis",
        "language": "Language"
    },
    "Hindi": {
        "title": "🩺 वैद्य AI - एक हेल्थकेयर असिस्टेंट",
        "subtitle": "एआई द्वारा संचालित पेशेवर चिकित्सा निदान",
        "input": "इनपुट",
        "voice_tab": "🎤 वॉइस इनपुट",
        "text_tab": "✍️ टेक्स्ट इनपुट",
        "describe_symptoms": "अपने लक्षणों का वर्णन करें",
        "earlier_symptoms": "पहले के लक्षण / आप किस प्रकार की समस्या का सामना कर रहे हैं?",
        "days_suffering": "कितने दिनों से",
        "days_help": "आप कितने दिनों से पीड़ित हैं?",
        "upload_image": "मेडिकल इमेज अपलोड करें (वैकल्पिक)",
        "doctor_panel": "आपके डॉक्टर",
        "get_diagnosis": "🔍 निदान प्राप्त करें",
        "language": "भाषा"
    },
    "Marathi": {
        "title": "🩺 वैद्य AI - आरोग्य सहाय्यक",
        "subtitle": "एआय द्वारा समर्थित व्यावसायिक वैद्यकीय निदान",
        "input": "इनपुट",
        "voice_tab": "🎤 आवाज इनपुट",
        "text_tab": "✍️ मजकूर इनपुट",
        "describe_symptoms": "आपली लक्षणे वर्णन करा",
        "earlier_symptoms": "पूर्वीची लक्षणे / तुम्ही कोणत्या प्रकारची समस्या अनुभवत आहात?",
        "days_suffering": "किती दिवसांपासून",
        "days_help": "आपण किती दिवसांपासून त्रस्त आहात?",
        "upload_image": "वैद्यकीय प्रतिमा अपलोड करा (ऐच्छिक)",
        "doctor_panel": "आपले डॉक्टर",
        "get_diagnosis": "🔍 निदान मिळवा",
        "language": "भाषा"
    }
}

def tr(key: str) -> str:
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)

# Top bar with language selector (top-right)
header_left, header_spacer, header_right = st.columns([8, 2, 2], gap="small")
with header_left:
    st.markdown(f"<h1 class='title-nowrap'>{tr('title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"*{tr('subtitle')}*")
with header_right:
    st.selectbox(tr("language"), list(LANGUAGE_CODES.keys()), key="language")

st.markdown("""
<style>
    .block-container {padding-top: 0.5rem; padding-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .stTextArea textarea {font-size: 1rem;}
    .diagnosis-card, .prescription-card {
        background: #22232b;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #fff;
        border: 1px solid #444;
    }
    .section-title {color: #ff9800; font-weight: bold; margin-bottom: 0.5rem;}
    .title-nowrap {white-space: nowrap; font-size: clamp(1.5rem, 2.6vw + 0.5rem, 3rem);}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown(f"### {tr('input')}")
    tab1, tab2 = st.tabs([tr("voice_tab"), tr("text_tab")])
    with tab1:
        st.caption("Upload an audio file for transcription")
        audio_input = st.file_uploader("Record your symptoms (upload .wav/.mp3)", type=["wav", "mp3"])
        if audio_input is None:
            st.info("Please upload an audio file or use text input")
        # Days suffering in Voice tab (synced to session state)
        st.number_input(tr("days_suffering"), min_value=0, step=1, help=tr("days_help"), key="duration_days_voice", value=st.session_state.get("duration_days_general", 0))
        if "duration_days_voice" in st.session_state:
            st.session_state["duration_days_general"] = st.session_state.get("duration_days_voice", 0)
    with tab2:
        c1, c2 = st.columns([3, 1])
        with c1:
            text_input = st.text_area(tr("describe_symptoms"), value=st.session_state.get("prefill_text", ""), placeholder="Type your symptoms here...", height=120)
        with c2:
            st.number_input(tr("days_suffering"), min_value=0, step=1, help=tr("days_help"), key="duration_days_general", value=st.session_state.get("duration_days_general", 0))
    earlier_symptoms = st.text_area(tr("earlier_symptoms"), placeholder="List early signs or describe the specific problem type...", height=100)
    image_input = st.file_uploader(tr("upload_image"), type=["jpg", "jpeg", "png", "webp"])
    response_language = st.session_state.get("language", "English")
    submit_btn = st.button(tr("get_diagnosis"), width="stretch")

with col2:
    st.markdown(f"### {tr('doctor_panel')}")
    st.image("portrait-3d-female-doctor[1].jpg", caption="Your Doctor", width="stretch")

# Output section
if submit_btn:
    if not GROQ_API_KEY:
        st.error("❌ Cannot proceed without API key. Please add GROQ_API_KEY to Streamlit secrets.")
    else:
        with st.spinner("Processing..."):
            try:
                # Audio input handling
                if audio_input is not None:
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_input.name)[-1])
                    temp_audio.write(audio_input.read())
                    temp_audio.close()
                    audio_path = temp_audio.name
                    st.info("🎤 Processing audio input...")
                    text_input = transcribe_with_groq(
                        stt_model="whisper-large-v3",
                        audio_filepath=audio_path,
                        GROQ_API_KEY=GROQ_API_KEY
                    )
                    os.remove(audio_path)
                    if text_input:
                        st.success(f"✅ Audio transcribed: {text_input[:100]}...")
                
                # Image input handling
                image_base64 = None
                if image_input is not None:
                    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_input.name)[-1])
                    temp_image.write(image_input.read())
                    temp_image.close()
                    image_base64 = encode_image(temp_image.name)
                    os.remove(temp_image.name)
                    st.success("🖼️ Image processed successfully")
                
                # Diagnosis logic
                diagnosis = None
                prescription = None
                audio_filepath = None
                language_code = LANGUAGE_CODES.get(response_language, "en")
                duration_val = st.session_state.get("duration_days_general", 0)
                
                if text_input and not image_base64:
                    st.info("🧠 Analyzing text input...")
                    enriched_text = (
                        f"Patient report:\n"
                        f"- Symptoms: {text_input}\n"
                        f"- Earlier symptoms/problem: {earlier_symptoms}\n"
                        f"- Duration (days): {duration_val}\n\n"
                        f"Task: Provide a detailed medical assessment including:\n"
                        f"1) Differential diagnosis with reasoning\n"
                        f"2) Most likely diagnosis\n"
                        f"3) Red flags and when to seek urgent care\n"
                        f"4) Home care advice\n"
                        f"5) Prescription-style suggestions (OTC where appropriate)."
                    )
                    diagnosis = analyze_text_query(enriched_text, response_language)
                    prescription = generate_prescription(diagnosis, response_language)
                elif image_base64:
                    st.info("🧠 Analyzing image...")
                    image_prompt = (
                        f"Analyze this image with the following patient context.\n\n"
                        f"Patient report:\n"
                        f"- Symptoms: {text_input or 'Not provided'}\n"
                        f"- Earlier symptoms/problem: {earlier_symptoms}\n"
                        f"- Duration (days): {duration_val}\n\n"
                        f"Task: Provide a detailed medical assessment including differential diagnosis, likely diagnosis, red flags, home care, and prescription-style suggestions."
                    )
                    diagnosis = analyze_image_with_query(image_prompt, image_base64, response_language)
                    prescription = generate_prescription(diagnosis, response_language)
                
                # Audio diagnosis and prescription
                audio_bytes = None
                if diagnosis and prescription:
                    full_text_for_audio = f"Diagnosis: {diagnosis}. Prescription: {prescription}"
                    audio_bytes = generate_audio_from_text(full_text_for_audio, language_code)
                    if audio_bytes:
                        st.success("🎧 Audio generated successfully")
                
                # Output UI
                st.markdown("---")
                st.markdown("## 📋 Diagnosis Results")
                st.markdown("<div class='section-title'>Your Input Summary</div>", unsafe_allow_html=True)
                st.text_area("Input Summary", value=str(text_input) if text_input else "Image analysis", height=80, disabled=True, label_visibility="collapsed")
                st.markdown("<div class='section-title'>🩺 Detailed Diagnosis</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='diagnosis-card'>{diagnosis or ''}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>💊 Prescription</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prescription-card'>{prescription or ''}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>🎧 Audio Diagnosis</div>", unsafe_allow_html=True)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.info("No audio available.")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.error("Please check your GROQ_API_KEY environment variable and try again.")
