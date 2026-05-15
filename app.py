import streamlit as st
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from groq import Groq
from deepgram import DeepgramClient
from deepgram import PrerecordedOptions
from deepgram import FileSource

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🛡",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
    linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827
    );
    color: white;
}

h1, h2, h3, h4, h5 {
    color: white;
}

div[data-testid="metric-container"] {
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}

.stTextArea textarea {
    background-color: #111827;
    color: white;
    border-radius: 12px;
}

.stProgress > div > div {
    background-color: red;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🛡 Sentinel AI")

st.subheader(
    "AI-Powered Real-Time Voice Fraud Detection Platform"
)

st.write("""
Advanced cybersecurity voice intelligence system capable of:

- Real-Time Scam Detection
- AI Behavioral Analysis
- Emotional Manipulation Detection
- Voice Stress Detection
- Threat Intelligence
- Social Engineering Detection
- Fraud Risk Scoring
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙ Sentinel AI")

mode = st.sidebar.radio(
    "Select Analysis Mode",
    [
        "Upload Recording",
        "Real-Time Analysis"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
Supported Audio Formats:
- WAV
- MP3
- M4A
""")

st.sidebar.markdown("---")

st.sidebar.success("🟢 Sentinel AI Active")

# =========================================================
# API CLIENTS
# =========================================================

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

deepgram = DeepgramClient(
    st.secrets["DEEPGRAM_API_KEY"]
)

# =========================================================
# AUDIO WAVEFORM
# =========================================================

def show_waveform(audio_path):

    y, sr = librosa.load(audio_path)

    fig, ax = plt.subplots(figsize=(12, 3))

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax
    )

    ax.set_title("Voice Activity Waveform")

    ax.set_xlabel("Time")

    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

# =========================================================
# TRANSCRIPTION
# =========================================================

def transcribe_audio(audio_path):

    with open(audio_path, "rb") as audio:

        buffer_data = audio.read()

    payload: FileSource = {
        "buffer": buffer_data
    }

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        detect_language=True
    )

    response = (
        deepgram.listen.prerecorded.v("1")
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

    words = transcript_data["words"]

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
                f"\n\n🔹 Speaker {speaker + 1}: "
            )

            current_speaker = speaker

        formatted_transcript += text + " "

    detected_language = (
        response["results"]
        .get("languages", ["Unknown"])[0]
    )

    speakers_count = len(
        set(
            word.get("speaker", 0)
            for word in words
        )
    )

    return (
        formatted_transcript,
        detected_language,
        speakers_count
    )

# =========================================================
# AI ANALYSIS
# =========================================================

def analyze_call(transcript):

    prompt = f"""
You are Sentinel AI,
an advanced AI cybersecurity voice intelligence system.

Analyze this phone conversation naturally.

Do NOT rely on keyword matching.

Understand:
- fraud intent
- emotional manipulation
- social engineering
- trust exploitation
- scam likelihood
- impersonation
- fear tactics
- urgency pressure
- deceptive communication
- psychological manipulation

Tasks:

1. Determine whether the conversation is:
   - Fraudulent
   - Suspicious
   - Genuine

2. Explain WHY.

3. Analyze:
   - behavioral patterns
   - emotional pressure
   - scam tactics
   - manipulation indicators
   - threat likelihood

4. Provide:
   - Fraud Confidence Score
   - Threat Level
   - Emotional Tone
   - Stress Level
   - AI Summary

Transcript:
{transcript}
"""

    completion = (
        groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    )

    return (
        completion.choices[0]
        .message.content
    )

# =========================================================
# RISK ENGINE
# =========================================================

def calculate_risk(ai_analysis):

    analysis_lower = ai_analysis.lower()

    risk_score = 10

    if (
        "fraudulent" in analysis_lower
        or "high risk" in analysis_lower
    ):
        risk_score += 60

    if (
        "scam" in analysis_lower
        or "fraud" in analysis_lower
    ):
        risk_score += 20

    if (
        "manipulation" in analysis_lower
        or "social engineering" in analysis_lower
    ):
        risk_score += 10

    if (
        "fear" in analysis_lower
        or "pressure" in analysis_lower
    ):
        risk_score += 10

    risk_score = min(risk_score, 100)

    if risk_score >= 75:
        risk_level = "HIGH"

    elif risk_score >= 40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return risk_score, risk_level

# =========================================================
# EMOTION ENGINE
# =========================================================

def detect_emotion(ai_analysis):

    analysis_lower = ai_analysis.lower()

    emotion = "Neutral"

    if (
        "fear" in analysis_lower
        or "panic" in analysis_lower
    ):
        emotion = "Fear"

    elif (
        "angry" in analysis_lower
        or "threat" in analysis_lower
    ):
        emotion = "Aggressive"

    elif (
        "positive" in analysis_lower
        or "friendly" in analysis_lower
    ):
        emotion = "Positive"

    return emotion

# =========================================================
# VOICE STRESS ANALYSIS
# =========================================================

def voice_stress_analysis(audio_path):

    y, sr = librosa.load(audio_path)

    energy = np.mean(librosa.feature.rms(y=y))

    if energy > 0.1:
        return "High Stress"

    elif energy > 0.05:
        return "Moderate Stress"

    else:
        return "Low Stress"

# =========================================================
# DASHBOARD
# =========================================================

def show_dashboard(
    transcript,
    language,
    speakers,
    ai_analysis,
    audio_path
):

    risk_score, risk_level = (
        calculate_risk(ai_analysis)
    )

    emotion = detect_emotion(ai_analysis)

    stress_level = voice_stress_analysis(
        audio_path
    )

    # =====================================================

    if risk_level == "HIGH":

        st.error(
            "🚨 HIGH RISK FRAUD DETECTED"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠ SUSPICIOUS ACTIVITY DETECTED"
        )

    else:

        st.success(
            "✅ CALL APPEARS SAFE"
        )

    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌐 Language",
            language.upper()
        )

    with col2:
        st.metric(
            "🎧 Speakers",
            speakers
        )

    with col3:
        st.metric(
            "🚨 Risk Score",
            f"{risk_score}%"
        )

    with col4:
        st.metric(
            "😊 Emotion",
            emotion
        )

    # =====================================================

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🧠 Stress Level",
            stress_level
        )

    with col2:
        st.metric(
            "📊 Threat Level",
            risk_level
        )

    # =====================================================

    st.subheader("📈 Threat Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Threat Intelligence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================

    st.subheader("🎵 Voice Activity")

    show_waveform(audio_path)

    # =====================================================

    st.subheader("⏱ AI Activity Timeline")

    timeline_events = [
        "🎙 Voice captured",
        "🧠 Speech transcribed",
        "🔍 Behavioral analysis completed",
        "📊 Threat intelligence generated",
        "🚨 Risk scoring completed",
        "🛡 Sentinel AI monitoring active"
    ]

    for event in timeline_events:
        st.write(event)

    # =====================================================

    st.subheader("📄 Call Transcript")

    st.text_area(
        "Conversation",
        transcript,
        height=350
    )

    # =====================================================

    st.subheader("🧠 AI Fraud Analysis")

    st.write(ai_analysis)

    # =====================================================

    st.subheader("📋 AI Executive Summary")

    summary_prompt = f"""
Summarize this AI fraud analysis
in 5 concise bullet points.

Analysis:
{ai_analysis}
"""

    summary_completion = (
        groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ]
        )
    )

    summary = (
        summary_completion
        .choices[0]
        .message.content
    )

    st.write(summary)

    # =====================================================

    report = f"""
=================================================
SENTINEL AI CYBERSECURITY REPORT
=================================================

Generated:
{datetime.now()}

=================================================

LANGUAGE:
{language}

=================================================

SPEAKERS:
{speakers}

=================================================

RISK SCORE:
{risk_score}/100

=================================================

THREAT LEVEL:
{risk_level}

=================================================

EMOTIONAL TONE:
{emotion}

=================================================

VOICE STRESS:
{stress_level}

=================================================

TRANSCRIPT:
{transcript}

=================================================

AI ANALYSIS:
{ai_analysis}

=================================================

AI SUMMARY:
{summary}
"""

    st.download_button(
        label="⬇ Download AI Report",
        data=report,
        file_name="sentinel_ai_report.txt",
        mime="text/plain"
    )

# =========================================================
# REAL-TIME ANALYSIS
# =========================================================

if mode == "Real-Time Analysis":

    st.subheader("🎙 Real-Time Voice Analysis")

    duration = st.slider(
        "Recording Duration (Seconds)",
        5,
        60,
        10
    )

    if st.button("🎤 Start Live Recording"):

        st.info(
            "🎙 Listening to microphone..."
        )

        sample_rate = 16000

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )

        sd.wait()

        st.success(
            "✅ Recording Completed"
        )

        temp_audio = "live_recording.wav"

        sf.write(
            temp_audio,
            recording,
            sample_rate
        )

        st.audio(temp_audio)

        try:

            with st.spinner(
                "🧠 Sentinel AI analyzing behavioral patterns..."
            ):

                (
                    transcript,
                    language,
                    speakers
                ) = transcribe_audio(
                    temp_audio
                )

                ai_analysis = analyze_call(
                    transcript
                )

            show_dashboard(
                transcript,
                language,
                speakers,
                ai_analysis,
                temp_audio
            )

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )

# =========================================================
# FILE UPLOAD ANALYSIS
# =========================================================

if mode == "Upload Recording":

    uploaded_file = st.file_uploader(
        "📂 Upload Call Recording",
        type=["wav", "mp3", "m4a"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            temp_audio = tmp_file.name

        st.success(
            "✅ Audio Uploaded Successfully"
        )

        try:

            with st.spinner(
                "🧠 Sentinel AI analyzing behavioral patterns..."
            ):

                (
                    transcript,
                    language,
                    speakers
                ) = transcribe_audio(
                    temp_audio
                )

                ai_analysis = analyze_call(
                    transcript
                )

            show_dashboard(
                transcript,
                language,
                speakers,
                ai_analysis,
                temp_audio
            )

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )

        finally:

            if os.path.exists(temp_audio):
                os.remove(temp_audio)
