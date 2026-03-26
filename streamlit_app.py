import uuid
from typing import Dict, Any

import pandas as pd
import streamlit as st

from audio_processor import VoiceFeatureExtractor
from config import ExtractionConfig, DEFAULT_CONFIG


st.set_page_config(
    page_title="Voice Genetics",
    layout="wide",
)


@st.cache_resource
def get_extractor() -> VoiceFeatureExtractor:
    return VoiceFeatureExtractor(DEFAULT_CONFIG)


def make_config(
    pitch_min_f0: float,
    pitch_max_f0: float,
    formant_max_frequency: float,
    formant_number: int,
    mfcc_number: int,
    min_duration_seconds: float,
    target_sample_rate: int,
) -> ExtractionConfig:
    return ExtractionConfig(
        pitch_min_f0=pitch_min_f0,
        pitch_max_f0=pitch_max_f0,
        pitch_unit="Hz",
        formant_max_frequency=formant_max_frequency,
        formant_number=formant_number,
        mfcc_number=mfcc_number,
        mfcc_use_energy=True,
        jitter_method="RAP",
        shimmer_method="APQ3",
        min_snr_db=20.0,
        min_duration_seconds=min_duration_seconds,
        target_sample_rate=target_sample_rate,
    )


def safe_round(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def flatten_result(result) -> Dict[str, Any]:
    rq = result.recording_quality
    features = result.features
    pitch = features.get("pitch", {})
    timbre = features.get("timbre", {})
    voice_quality = features.get("voice_quality", {})

    flat = {
        "session_id": result.session_id,
        "processing_timestamp": result.processing_timestamp,
        "duration_seconds": safe_round(rq.get("duration_seconds", 0.0)),
        "snr_db": safe_round(rq.get("snr_db", 0.0)),
        "background_noise_level": rq.get("background_noise_level", "unknown"),
        "sample_rate": rq.get("sample_rate", 0),
        "mean_f0_hz": safe_round(pitch.get("mean_f0_hz", 0.0)),
        "min_f0_hz": safe_round(pitch.get("min_f0_hz", 0.0)),
        "max_f0_hz": safe_round(pitch.get("max_f0_hz", 0.0)),
        "variability": safe_round(pitch.get("variability", 0.0)),
        "jitter_percent": safe_round(voice_quality.get("jitter_percent", 0.0)),
        "shimmer_db": safe_round(voice_quality.get("shimmer_db", 0.0)),
        "harmonic_to_noise_ratio": safe_round(voice_quality.get("harmonic_to_noise_ratio", 0.0)),
    }

    formants = timbre.get("formants", {})
    for key, value in formants.items():
        flat[key] = safe_round(value)

    mfccs = timbre.get("mfccs", [])
    for idx, value in enumerate(mfccs, start=1):
        flat[f"mfcc_{idx}"] = safe_round(value)

    return flat


def display_overview(result) -> None:
    rq = result.recording_quality
    pitch = result.features.get("pitch", {})
    voice_quality = result.features.get("voice_quality", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration (s)", safe_round(rq.get("duration_seconds", 0.0)))
    c2.metric("SNR (dB)", safe_round(rq.get("snr_db", 0.0)))
    c3.metric("Mean F0 (Hz)", safe_round(pitch.get("mean_f0_hz", 0.0)))
    c4.metric("Noise level", rq.get("background_noise_level", "unknown"))

    c5, c6, c7 = st.columns(3)
    c5.metric("Jitter (%)", safe_round(voice_quality.get("jitter_percent", 0.0)))
    c6.metric("Shimmer (dB)", safe_round(voice_quality.get("shimmer_db", 0.0)))
    c7.metric("HNR", safe_round(voice_quality.get("harmonic_to_noise_ratio", 0.0)))


def display_pitch(result) -> None:
    pitch = result.features.get("pitch", {})
    df = pd.DataFrame(
        {
            "Metric": ["Mean F0", "Min F0", "Max F0", "Variability"],
            "Value": [
                safe_round(pitch.get("mean_f0_hz", 0.0)),
                safe_round(pitch.get("min_f0_hz", 0.0)),
                safe_round(pitch.get("max_f0_hz", 0.0)),
                safe_round(pitch.get("variability", 0.0)),
            ],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_formants(result) -> None:
    formants = result.features.get("timbre", {}).get("formants", {})
    if not formants:
        st.info("No formants were extracted.")
        return

    df = pd.DataFrame(
        {
            "Formant": list(formants.keys()),
            "Frequency (Hz)": [safe_round(v) for v in formants.values()],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Formant"))


def display_mfccs(result) -> None:
    mfccs = result.features.get("timbre", {}).get("mfccs", [])
    if not mfccs:
        st.info("No MFCCs were extracted.")
        return

    df = pd.DataFrame(
        {
            "Coefficient": [f"MFCC {i}" for i in range(1, len(mfccs) + 1)],
            "Value": [safe_round(v) for v in mfccs],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.line_chart(df.set_index("Coefficient"))


def display_quality(result) -> None:
    rq = result.recording_quality
    voice_quality = result.features.get("voice_quality", {})

    left, right = st.columns(2)

    with left:
        st.subheader("Recording quality")
        st.json(
            {
                "duration_seconds": safe_round(rq.get("duration_seconds", 0.0)),
                "snr_db": safe_round(rq.get("snr_db", 0.0)),
                "background_noise_level": rq.get("background_noise_level", "unknown"),
                "sample_rate": rq.get("sample_rate", 0),
            }
        )

    with right:
        st.subheader("Voice quality")
        st.json(
            {
                "jitter_percent": safe_round(voice_quality.get("jitter_percent", 0.0)),
                "shimmer_db": safe_round(voice_quality.get("shimmer_db", 0.0)),
                "harmonic_to_noise_ratio": safe_round(voice_quality.get("harmonic_to_noise_ratio", 0.0)),
            }
        )


def add_result_to_history(result) -> None:
    flat = flatten_result(result)
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, flat)


def show_history() -> None:
    st.subheader("Recent analyses")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No analyses yet.")
        return

    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True)
    st.download_button(
        label="Download history as CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="voice_genetics_history.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("🎵 Voice Genetics")
    st.markdown(
        "Upload a voice recording and extract acoustic features such as pitch, formants, MFCCs, "
        "jitter, shimmer, and basic recording quality metrics."
    )

    with st.sidebar:
        st.header("Settings")
        pitch_min_f0 = st.number_input("Min pitch (Hz)", min_value=50.0, max_value=500.0, value=75.0, step=1.0)
        pitch_max_f0 = st.number_input("Max pitch (Hz)", min_value=100.0, max_value=1000.0, value=300.0, step=1.0)
        formant_max_frequency = st.number_input(
            "Formant max frequency (Hz)", min_value=1000.0, max_value=10000.0, value=5500.0, step=100.0
        )
        formant_number = st.slider("Number of formants", min_value=1, max_value=6, value=4)
        mfcc_number = st.slider("Number of MFCCs", min_value=5, max_value=30, value=13)
        min_duration_seconds = st.number_input(
            "Minimum duration (s)", min_value=0.1, max_value=30.0, value=0.5, step=0.1
        )
        target_sample_rate = st.selectbox("Target sample rate", options=[8000, 16000, 22050, 44100], index=1)

        st.markdown("---")
        st.caption("Supported formats: WAV, MP3, M4A")

    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

    analyze = st.button("Analyze voice", type="primary", use_container_width=True)

    if analyze:
        if uploaded_file is None:
            st.error("Please upload an audio file first.")
        else:
            try:
                audio_bytes = uploaded_file.read()
                if len(audio_bytes) > 50 * 1024 * 1024:
                    st.error("File too large. Maximum size is 50MB.")
                    st.stop()

                config = make_config(
                    pitch_min_f0=pitch_min_f0,
                    pitch_max_f0=pitch_max_f0,
                    formant_max_frequency=formant_max_frequency,
                    formant_number=formant_number,
                    mfcc_number=mfcc_number,
                    min_duration_seconds=min_duration_seconds,
                    target_sample_rate=target_sample_rate,
                )

                extractor = get_extractor()
                extractor.config = config

                session_id = str(uuid.uuid4())
                with st.spinner("Processing audio and extracting features..."):
                    result = extractor.extract_features(audio_bytes, session_id)

                actual_duration = result.recording_quality.get("duration_seconds", 0.0)
                if actual_duration < min_duration_seconds:
                    st.warning(
                        f"Audio is too short. Minimum duration is {min_duration_seconds}s, but the uploaded file is {safe_round(actual_duration)}s."
                    )
                else:
                    st.success("Feature extraction completed successfully.")

                st.subheader("Overview")
                display_overview(result)

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["Pitch", "Formants", "MFCCs", "Quality", "Raw JSON"]
                )

                with tab1:
                    display_pitch(result)

                with tab2:
                    display_formants(result)

                with tab3:
                    display_mfccs(result)

                with tab4:
                    display_quality(result)

                with tab5:
                    st.json(
                        {
                            "session_id": result.session_id,
                            "recording_quality": result.recording_quality,
                            "features": result.features,
                            "processing_timestamp": result.processing_timestamp,
                        }
                    )

                add_result_to_history(result)

                json_payload = {
                    "session_id": result.session_id,
                    "recording_quality": result.recording_quality,
                    "features": result.features,
                    "processing_timestamp": result.processing_timestamp,
                }
                st.download_button(
                    label="Download result as JSON",
                    data=pd.Series(json_payload).to_json(indent=2),
                    file_name=f"voice_features_{result.session_id}.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Feature extraction failed: {e}")

    st.markdown("---")
    show_history()

    with st.expander("How to run this app"):
        st.code(
            "pip install -r requirements.txt\n"
            "pip install streamlit\n"
            "streamlit run streamlit_app.py",
            language="bash",
        )


if __name__ == "__main__":
    main()
