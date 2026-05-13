import streamlit as st
import whisper
import tempfile
import os

from groq import Groq
from transformers import pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Fraud Call Detector",
    page_icon="📞",
    layout="wide"
)

st.title("📞 AI Fraud Call Detection System")

st.markdown(
    """
    Upload a phone call recording and let AI detect:
    - Fraud / Scam calls
    - Spam calls
    - Genuine conversations
    - Emotional tone
    """
)

# -----------------------------
# GROQ API
# -----------------------------

api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# -----------------------------
# LOAD MODELS
# -----------------------------

@st.cache_resource
def load_models():

    # Better Hindi + English understanding
    whisper_model = whisper.load_model("medium")

    emotion_model = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion"
    )

    return whisper_model, emotion_model


model, emotion_classifier = load_models()

# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Audio File",
    type=["wav", "mp3", "m4a"]
)

# -----------------------------
# PROCESS AUDIO
# -----------------------------

if uploaded_file is not None:

    # SAFER TEMP FILE FOR STREAMLIT CLOUD
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_audio = tmp_file.name

    st.success("✅ Audio Uploaded Successfully!")

    # Audio Player
    st.audio(uploaded_file)

    with st.spinner("🔍 Analyzing Call..."):

        try:

            # -----------------------------
            # TRANSCRIPTION
            # -----------------------------

            result = model.transcribe(
                temp_audio,
                language=None,
                task="transcribe",
                fp16=False,
                temperature=0.2,
                best_of=5
            )

            transcript_text = result["text"]

            detected_language = result["language"]

            # -----------------------------
            # DETECTED LANGUAGE
            # -----------------------------

            st.subheader("🌐 Detected Language")

            st.write(detected_language.upper())

            # -----------------------------
            # SPEAKER FORMATTING
            # -----------------------------

            segments = result["segments"]

            formatted_transcript = ""

            speaker = 1

            for segment in segments:

                text = segment["text"].strip()

                if text:

                    formatted_transcript += (
                        f"Speaker {speaker}: {text}\n"
                    )

                    # Alternate speakers
                    speaker = 2 if speaker == 1 else 1

            # -----------------------------
            # TRANSCRIPT DISPLAY
            # -----------------------------

            st.subheader("📄 Transcript")

            st.text_area(
                "Conversation Transcript",
                formatted_transcript,
                height=400
            )

            # -----------------------------
            # AI FRAUD ANALYSIS
            # -----------------------------

            prompt = f"""
You are an AI fraud detection expert.

Analyze this Hindi-English phone conversation carefully.

Tasks:

1. Detect whether the call is:
   - Fraud / Scam
   - Spam
   - Genuine

2. Explain WHY.

3. Detect scam indicators such as:
   - OTP requests
   - Bank fraud
   - UPI scams
   - Threats
   - Urgency
   - Prize scams
   - Fake customer support
   - Identity theft
   - Financial manipulation

4. Give:
   - Fraud confidence score (0-100)
   - Short summary
   - Risk level (Low / Medium / High)

Conversation:
{formatted_transcript}
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            ai_analysis = response.choices[0].message.content

            # -----------------------------
            # DISPLAY AI ANALYSIS
            # -----------------------------

            st.subheader("📊 AI Fraud Analysis")

            st.write(ai_analysis)

            # -----------------------------
            # FRAUD DETECTION
            # -----------------------------

            if (
                "fraud" in ai_analysis.lower()
                or "scam" in ai_analysis.lower()
                or "spam" in ai_analysis.lower()
            ):

                st.error("⚠ Potential Fraud / Scam Call Detected")

                risk_score = 85

            else:

                st.success("✅ Genuine Conversation Detected")

                risk_score = 20

            # -----------------------------
            # RISK METER
            # -----------------------------

            st.subheader("🚨 Risk Meter")

            st.progress(risk_score)

            st.write(f"Risk Score: {risk_score}/100")

            # -----------------------------
            # EMOTION DETECTION
            # -----------------------------

            emotion_result = emotion_classifier(
                transcript_text[:1000]
            )

            emotion = emotion_result[0]["label"]

            confidence = emotion_result[0]["score"]

            # -----------------------------
            # DISPLAY EMOTION
            # -----------------------------

            st.subheader("😊 Emotion Detection")

            st.write(f"Emotion: {emotion}")

            st.write(
                f"Confidence: {round(confidence, 2)}"
            )

            # -----------------------------
            # DOWNLOAD REPORT
            # -----------------------------

            report = f"""
========== DETECTED LANGUAGE ==========

{detected_language}

========== TRANSCRIPT ==========

{formatted_transcript}

========== FRAUD ANALYSIS ==========

{ai_analysis}

========== EMOTION DETECTION ==========

Emotion: {emotion}

Confidence: {round(confidence, 2)}

========== RISK SCORE ==========

{risk_score}/100
"""

            st.download_button(
                label="⬇ Download Full Report",
                data=report,
                file_name="call_analysis_report.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")

        finally:

            # CLEANUP TEMP FILE
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
