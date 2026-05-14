import streamlit as st
import tempfile
import os

from groq import Groq

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

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
# API CLIENTS
# -----------------------------

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

deepgram = DeepgramClient(
    st.secrets["DEEPGRAM_API_KEY"]
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

        # -----------------------------
        # TRANSCRIPTION + DIARIZATION
        # -----------------------------

        with st.spinner("🔍 Transcribing Audio..."):

            with open(temp_audio, "rb") as file:

                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                diarize=True,
                detect_language=True,
            )

            response = (
                deepgram.listen.prerecorded
                .v("1")
                .transcribe_file(
                    payload,
                    options
                )
            )

            transcript_data = (
                response["results"]
                ["channels"][0]
                ["alternatives"][0]
            )

            transcript_text = (
                transcript_data["transcript"]
            )

            words = transcript_data["words"]

        # -----------------------------
        # LANGUAGE DETECTION
        # -----------------------------

        detected_language = (
            response["results"]
            .get("languages", ["unknown"])[0]
        )

        st.subheader("🌐 Detected Language")

        st.write(detected_language.upper())

        # -----------------------------
        # REAL SPEAKER DETECTION
        # -----------------------------

        formatted_transcript = ""

        current_speaker = None

        for word in words:

            speaker = word.get("speaker", 0)

            text = word.get(
                "punctuated_word",
                word["word"]
            )

            if speaker != current_speaker:

                formatted_transcript += (
                    f"\n\nSpeaker {speaker + 1}: "
                )

                current_speaker = speaker

            formatted_transcript += text + " "

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
        # FRAUD ANALYSIS
        # -----------------------------

        with st.spinner("🧠 Detecting Fraud..."):

            prompt = f"""
You are an advanced fraud detection AI.

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

            response_ai = (
                groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            )

            ai_analysis = (
                response_ai.choices[0]
                .message.content
            )

        # -----------------------------
        # DISPLAY ANALYSIS
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
        # RISK METER
        # -----------------------------

        st.subheader("🚨 Risk Meter")

        st.progress(risk_score)

        st.write(
            f"Risk Score: {risk_score}/100"
        )

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

        st.write(
            f"Detected Emotion: {emotion}"
        )

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
