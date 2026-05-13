import streamlit as st
import whisper
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

# -----------------------------
# GROQ API
# -----------------------------

client = Groq(
    api_key=st.secrets["ZimfPn7eN7ZZK6EnDHpNWGdyb3FY2N3IyJEljtkZMMf64ia9QPSc"]
)

# -----------------------------
# LOAD MODELS
# -----------------------------

@st.cache_resource
def load_models():

    whisper_model = whisper.load_model("base")

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
    "Upload Audio File",
    type=["wav", "mp3", "m4a"]
)

# -----------------------------
# PROCESS AUDIO
# -----------------------------

if uploaded_file is not None:

    temp_audio = "temp_audio.wav"

    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Audio Uploaded Successfully!")

    with st.spinner("Analyzing Call..."):

        # -----------------------------
        # TRANSCRIPTION
        # -----------------------------

        result = model.transcribe(temp_audio)

        transcript_text = result["text"]

        # -----------------------------
        # FAKE SPEAKER SPLITTING
        # -----------------------------

        sentences = transcript_text.split(".")

        formatted_transcript = ""

        speaker = 1

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:

                formatted_transcript += f"Speaker {speaker}: {sentence}\n"

                # Alternate speakers
                speaker = 2 if speaker == 1 else 1

        # -----------------------------
        # DISPLAY TRANSCRIPT
        # -----------------------------

        st.subheader("📄 Transcript")

        st.text_area(
            "Conversation",
            formatted_transcript,
            height=400
        )

        # -----------------------------
        # GROQ FRAUD ANALYSIS
        # -----------------------------

        prompt = f"""
        Analyze the following call transcript.

        Detect:
        1. Is this call Fraud/Spam or Genuine?
        2. Explain why.
        3. Give a short summary.
        4. Mention suspicious behavior if any.

        Transcript:
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

        ai_analysis = response.choices[0].message.content

        # -----------------------------
        # EMOTION DETECTION
        # -----------------------------

        emotion_result = emotion_classifier(
            transcript_text[:1000]
        )

        emotion = emotion_result[0]["label"]
        confidence = emotion_result[0]["score"]

        # -----------------------------
        # SHOW RESULTS
        # -----------------------------

        st.subheader("📊 Fraud Analysis")

        st.write(ai_analysis)

        st.subheader("😊 Emotion Detection")

        st.write(f"Emotion: {emotion}")
        st.write(f"Confidence: {round(confidence, 2)}")

        # -----------------------------
        # DOWNLOAD REPORT
        # -----------------------------

        report = f"""
========== TRANSCRIPT ==========

{formatted_transcript}

========== FRAUD ANALYSIS ==========

{ai_analysis}

========== EMOTION ==========

Emotion: {emotion}

Confidence: {round(confidence, 2)}
"""

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name="call_report.txt",
            mime="text/plain"
        )