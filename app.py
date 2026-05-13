import streamlit as st
import tempfile
import os

from groq import Groq

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Fraud Call Detector",
    page_icon="📞",
    layout="wide"
)

st.title("📞 AI Fraud Call Detection System")

st.write(
    "Upload a call recording to detect fraud, scams, spam, and emotional tone."
)

# -----------------------------
# GROQ API
# -----------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

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

    st.audio(uploaded_file)

    # Save temporary audio
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_audio = tmp_file.name

    st.success("✅ Audio Uploaded Successfully!")

    try:

        with st.spinner("🔍 Transcribing Audio..."):

            # -----------------------------
            # AUDIO TRANSCRIPTION USING GROQ
            # -----------------------------

            with open(temp_audio, "rb") as audio_file:

                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    response_format="verbose_json"
                )

            transcript_text = transcription.text

        # -----------------------------
        # LANGUAGE DETECTION
        # -----------------------------

        detected_language = getattr(
            transcription,
            "language",
            "unknown"
        )

        st.subheader("🌐 Detected Language")

        st.write(detected_language.upper())

        # -----------------------------
        # SPEAKER FORMATTING
        # -----------------------------

        sentences = transcript_text.split(".")

        formatted_transcript = ""

        speaker = 1

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:

                formatted_transcript += (
                    f"Speaker {speaker}: {sentence}\n"
                )

                speaker = 2 if speaker == 1 else 1

        # -----------------------------
        # TRANSCRIPT
        # -----------------------------

        st.subheader("📄 Transcript")

        st.text_area(
            "Conversation Transcript",
            formatted_transcript,
            height=350
        )

        # -----------------------------
        # FRAUD ANALYSIS
        # -----------------------------

        with st.spinner("🧠 Detecting Fraud..."):

            prompt = f"""
You are an advanced fraud detection AI.

Analyze this Hindi-English phone conversation.

Detect:
1. Fraud / Scam / Genuine
2. Scam indicators
3. Fraud confidence score
4. Risk level
5. Short summary

Look for:
- OTP scams
- UPI fraud
- Banking fraud
- Fake customer care
- Threats
- Urgency
- Money requests
- Prize scams
- Identity theft

Conversation:
{formatted_transcript}
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            ai_analysis = (
                response.choices[0]
                .message.content
            )

        # -----------------------------
        # SHOW ANALYSIS
        # -----------------------------

        st.subheader("📊 AI Fraud Analysis")

        st.write(ai_analysis)

        # -----------------------------
        # RISK DETECTION
        # -----------------------------

        analysis_lower = ai_analysis.lower()

        if (
            "fraud" in analysis_lower
            or "scam" in analysis_lower
            or "spam" in analysis_lower
        ):

            risk_score = 85

            st.error(
                "⚠ Potential Fraud / Scam Call Detected"
            )

        else:

            risk_score = 20

            st.success(
                "✅ Genuine Conversation Detected"
            )

        # -----------------------------
        # RISK BAR
        # -----------------------------

        st.subheader("🚨 Risk Meter")

        st.progress(risk_score)

        st.write(f"Risk Score: {risk_score}/100")

        # -----------------------------
        # SIMPLE EMOTION DETECTION
        # -----------------------------

        emotion = "Neutral"

        if (
            "angry" in analysis_lower
            or "threat" in analysis_lower
        ):
            emotion = "Angry"

        elif (
            "fear" in analysis_lower
            or "panic" in analysis_lower
        ):
            emotion = "Fear"

        elif (
            "happy" in analysis_lower
            or "excited" in analysis_lower
        ):
            emotion = "Happy"

        st.subheader("😊 Emotion Detection")

        st.write(f"Detected Emotion: {emotion}")

        # -----------------------------
        # REPORT DOWNLOAD
        # -----------------------------

        report = f"""
========== LANGUAGE ==========

{detected_language}

========== TRANSCRIPT ==========

{formatted_transcript}

========== FRAUD ANALYSIS ==========

{ai_analysis}

========== EMOTION ==========

{emotion}

========== RISK SCORE ==========

{risk_score}/100
"""

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name="fraud_analysis_report.txt",
            mime="text/plain"
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")

    finally:

        if os.path.exists(temp_audio):
            os.remove(temp_audio)
