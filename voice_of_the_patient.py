# Simplified voice module for Streamlit deployment
import logging
import os
# --- Make Groq optional (so app can still run without voice on Streamlit Cloud) ---
try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None  # type: ignore
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key():
    """Get API key from Streamlit secrets or environment variables"""
    try:
        import streamlit as st
        # Streamlit Cloud: secrets are preferred
        return str(st.secrets.get("GROQ_API_KEY")) if st.secrets.get("GROQ_API_KEY") else os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY=None):
    """Transcribe audio using Groq API with Whisper model"""
    logger.info(f"Attempting to transcribe {audio_filepath} with model {stt_model}")
    try:
        if Groq is None:
            raise RuntimeError("groq package not available")

        api_key = GROQ_API_KEY or get_api_key()
        if not api_key:
            raise ValueError("Groq API key not found.")

        client = Groq(api_key=api_key)

        with open(audio_filepath, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_filepath), file.read()),
                model=stt_model,
            )

        logger.info("Transcription successful.")
        return transcription.text

    except Exception as e:
        logger.error(f"Error transcribing with Groq: {str(e)}")
        if "401" in str(e) or "invalid_api_key" in str(e).lower():
            logger.error("The provided Groq API key is invalid or expired.")
        elif "403" in str(e):
            logger.error("Insufficient permissions or quota on Groq.")
        elif "429" in str(e):
            logger.error("Rate limit exceeded for Groq API.")
        return None

def record_audio(file_path):
    """
    Placeholder for audio recording - simplified for Streamlit
    """
    try:
        import streamlit as st
        
        # Use Streamlit's audio recorder
        audio_bytes = st.audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            sample_rate=16000
        )

        if audio_bytes is not None:
            # Save the audio bytes directly
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Audio recorded and saved to {file_path}")
            return True
        else:
            logger.warning("No audio was recorded")
            return False

    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        return False

def transcribe_audio(file_path):
    """Legacy placeholder (kept for backward compatibility).

    The Streamlit app uses Groq Whisper via `transcribe_with_groq`.
    This function is intentionally a no-op to avoid requiring `speech_recognition`.
    """
    return None

def main():
    try:
        import streamlit as st
        st.title("Voice of the Patient")
        
        # Create a directory for audio files if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        
        # Record audio
        if st.button("Start Recording"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join("recordings", f"recording_{timestamp}.wav")
            
            if record_audio(file_path):
                st.success("Recording completed!")
                
                # Transcribe the audio
                text = transcribe_audio(file_path)
                if text:
                    st.write("Transcription:", text)
                    
                    # Process with Groq
                    response = transcribe_with_groq("whisper-large-v3", file_path)
                    if response:
                        st.write("AI Analysis:", response)
                    else:
                        st.error("Failed to process with AI")
                else:
                    st.error("Failed to transcribe the audio")
            else:
                st.error("Failed to record audio")
    except ImportError:
        print("Streamlit not available for main function")

if __name__ == "__main__":
    main()
