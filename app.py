
import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
from pathlib import Path

# Load the trained model
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_music_genre_v2.pkl"

model = joblib.load(MODEL_PATH)

# Feature extraction function
def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=22050,
        mono=True,
        duration=30
    )

    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    # 3. Spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)

    # 4. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    # 5. RMS energy
    rms = librosa.feature.rms(y=audio)

    # 6. Tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    # 7. Tonnetz
    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    # Combine features in EXACT training order
    feature_row = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),

        np.mean(centroid),
        np.std(centroid),

        np.mean(bandwidth),
        np.std(bandwidth),

        np.mean(rolloff),
        np.std(rolloff),

        np.mean(zcr),
        np.std(zcr),

        np.mean(contrast, axis=1),
        np.std(contrast, axis=1),

        np.mean(rms),
        np.std(rms),

        float(np.mean(tempo)),

        np.mean(tonnetz, axis=1),
        np.std(tonnetz, axis=1)
    ])

    return feature_row.reshape(1, -1)


st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .stApp {
        background-color: #100E19;
        color: #F4F0FF;
    }

    h1, h2, h3 {
        color: #F4F0FF !important;
    }

    p, label, .stMarkdown {
        color: #C9C1DC;
    }

    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stFileUploader"] {
        background: #1D192B;
        border: 1px solid #44365F;
        border-radius: 16px;
        padding: 20px;
    }

    div.stButton > button {
        background: #9875E8;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #B39AF5;
        color: #100E19;
    }
</style>
""", unsafe_allow_html=True)

st.title(" Music Genre Studio")
st.write("Discover the sound behind your music.")


# Audio upload
st.markdown("### Analyse Your Track")
st.caption("Upload an audio file to identify its musical genre.")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac"],
    help="Supported formats: WAV, MP3, OGG and FLAC."
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button(" Predict Genre", type="primary"):

        temp_path = None

        try:
            # Save uploaded audio temporarily
            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            with st.spinner("Analyzing your music..."):

                features = extract_features(temp_path)

                prediction = model.predict(features)[0]

            st.markdown(
                f"""
                <div style="background-color:#1D192B;
                            border:1px solid #7657B5;
                            border-radius:16px;
                            padding:28px;
                            margin-top:24px;">
                    <div style="color:#A995D8;
                                font-size:13px;
                                letter-spacing:2px;
                                margin-bottom:12px;">
                        ANALYSIS COMPLETE
                    </div>
                    <div style="color:#C9B5FF;
                                font-size:16px;
                                margin-bottom:8px;">
                        Predicted Music Genre
                    </div>
                    <div style="color:#F4F0FF;
                                font-size:42px;
                                font-weight:700;">
                        {str(prediction).title()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Could not analyze this audio file: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.markdown("---")

st.markdown("## About the Model")

st.write(
    "Music Genre Studio uses a machine learning model to identify "
    "the genre of an uploaded audio track. It extracts 87 audio "
    "features using Librosa and uses a Support Vector Machine (SVM) "
    "classifier to predict the genre."
)

st.markdown("### How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**01 — Upload**")
    st.caption("Choose an audio file.")

with col2:
    st.markdown("**02 — Feature Extraction**")
    st.caption("Extract musical and spectral characteristics.")

with col3:
    st.markdown("**03 — Prediction**")
    st.caption("The trained SVM model predicts the genre.")
st.markdown("---")
st.caption("Powered by Librosa, Python and Machine Learning")
