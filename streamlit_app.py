import json
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


METRIC_EXPLANATIONS = {
    "Duration (s)": "Length of the uploaded recording in seconds.",
    "SNR (dB)": "Signal-to-noise ratio. Higher values usually mean cleaner audio with less background noise.",
    "Average pitch (Hz)": "Average fundamental frequency of the voice, often perceived as average pitch.",
    "Noise level": "A simple estimated category of background noise in the recording.",
    "Jitter (%)": "Small cycle-to-cycle changes in pitch. Higher values may indicate less stable voice production.",
    "Shimmer (dB)": "Small cycle-to-cycle changes in loudness. Higher values may indicate less stable vocal intensity.",
    "Voice clarity (HNR)": "Harmonic-to-noise ratio. Higher values usually suggest a clearer, more periodic voice signal.",
    "Min F0": "Lowest detected pitch in the voiced parts of the recording.",
    "Max F0": "Highest detected pitch in the voiced parts of the recording.",
    "Variability": "How much the pitch changes relative to its average value.",
    "Formants": "Resonance frequencies of the vocal tract. They help describe how speech sounds are shaped.",
    "MFCCs": "Mel-frequency cepstral coefficients. These summarize the sound spectrum and are commonly used in speech analysis and machine learning.",
    "Sample rate": "Number of audio samples per second used during processing.",
}


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


def metric_with_help(title: str, value: Any, help_text: str | None = None) -> None:
    st.metric(title, value)
    if help_text:
        st.caption(help_text)


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
    with c1:
        metric_with_help(
            "Duration (s)",
            safe_round(rq.get("duration_seconds", 0.0)),
            METRIC_EXPLANATIONS["Duration (s)"],
        )
    with c2:
        metric_with_help(
            "SNR (dB)",
            safe_round(rq.get("snr_db", 0.0)),
            METRIC_EXPLANATIONS["SNR (dB)"],
        )
    with c3:
        metric_with_help(
            "Average pitch (Hz)",
            safe_round(pitch.get("mean_f0_hz", 0.0)),
            METRIC_EXPLANATIONS["Average pitch (Hz)"],
        )
    with c4:
        metric_with_help(
            "Noise level",
            rq.get("background_noise_level", "unknown"),
            METRIC_EXPLANATIONS["Noise level"],
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        metric_with_help(
            "Jitter (%)",
            safe_round(voice_quality.get("jitter_percent", 0.0)),
            METRIC_EXPLANATIONS["Jitter (%)"],
        )
    with c6:
        metric_with_help(
            "Shimmer (dB)",
            safe_round(voice_quality.get("shimmer_db", 0.0)),
            METRIC_EXPLANATIONS["Shimmer (dB)"],
        )
    with c7:
        metric_with_help(
            "Voice clarity (HNR)",
            safe_round(voice_quality.get("harmonic_to_noise_ratio", 0.0)),
            METRIC_EXPLANATIONS["Voice clarity (HNR)"],
        )

    st.info(
        "These values describe recording quality, pitch behavior, and voice stability. "
        "They are useful for analysis, but they are not a medical diagnosis on their own."
    )


def display_pitch(result) -> None:
    st.caption(
        "Pitch features describe how high or low the voice sounds and how much that pitch changes."
    )

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
            "Meaning": [
                METRIC_EXPLANATIONS["Average pitch (Hz)"],
                METRIC_EXPLANATIONS["Min F0"],
                METRIC_EXPLANATIONS["Max F0"],
                METRIC_EXPLANATIONS["Variability"],
            ],
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_formants(result) -> None:
    st.caption(
        "Formants are resonance frequencies of the vocal tract. They help describe how speech sounds are shaped."
    )

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
    st.caption(
        "MFCCs summarize the spectral shape of the voice signal and are commonly used as machine-learning input features."
    )

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
    st.caption(
        "These metrics describe the quality of the recording itself and the stability of the produced voice signal."
    )

    rq = result.recording_quality
    voice_quality = result.features.get("voice_quality", {})

    left, right = st.columns(2)

    with left:
        st.subheader("Recording quality")
        st.write("How clean and usable the uploaded audio is.")
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
        st.write("How stable and periodic the voice signal appears to be.")
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
    st.title("Voice Genetics")
    st.markdown(
        "Upload a voice recording and extract acoustic features such as pitch, formants, MFCCs, "
        "jitter, shimmer, and basic recording quality metrics."
    )

    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
            - **Duration**: the length of the recording.
            - **SNR**: how strong the voice signal is compared to background noise.
            - **Average pitch (F0)**: the average perceived pitch of the voice.
            - **Noise level**: a simple estimate of background noise.
            - **Jitter**: small pitch instability from one cycle to the next.
            - **Shimmer**: small loudness instability from one cycle to the next.
            - **HNR**: ratio of harmonic voice energy to noise energy.
            - **Formants**: resonance frequencies linked to vocal tract shape.
            - **MFCCs**: compact features describing the sound spectrum.

            These measurements help describe the voice signal, but they should not be interpreted as a diagnosis by themselves.
            """
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
                        f"Audio is too short. Minimum duration is {min_duration_seconds}s, "
                        f"but the uploaded file is {safe_round(actual_duration)}s."
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
                    data=json.dumps(json_payload, indent=2),
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
            "python -m streamlit run streamlit_app.py",
            language="bash",
        )


if __name__ == "__main__":
    main()