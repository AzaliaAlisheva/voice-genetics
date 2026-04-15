import json
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

import pandas as pd
import streamlit as st

from audio_processor import VoiceFeatureExtractor
from config import ExtractionConfig, DEFAULT_CONFIG
from dialogue_processor import DialogueProcessor, DialogueSegmentSpec
from genetic_model import predictor

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Voice Genetics",
    layout="wide",
)

# Dictionary of human-readable explanations for each metric shown in the UI
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
    """Cached extractor instance to avoid reloading on every rerun."""
    return VoiceFeatureExtractor(DEFAULT_CONFIG)


def make_config(
        pitch_min_f0: float,
        pitch_max_f0: float,
        formant_max_frequency: float,
        formant_number: int,
        mfcc_number: int,
        min_duration_seconds: float,
        target_sample_rate: int,
        vad_top_db: float = 28.0,
        min_turn_duration_seconds: float = 0.35,
        merge_gap_seconds: float = 0.25,
) -> ExtractionConfig:
    """Create an ExtractionConfig from user-provided parameters."""
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
        vad_top_db=vad_top_db,
        min_turn_duration_seconds=min_turn_duration_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )


def safe_round(value: Any, digits: int = 3) -> Any:
    """Round numeric values safely; return non-numeric values unchanged."""
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def metric_with_help(title: str, value: Any, help_text: str | None = None) -> None:
    """Display a metric with an optional help caption explaining what it means."""
    st.metric(title, value)
    if help_text:
        st.caption(help_text)


def flatten_result(result) -> Dict[str, Any]:
    """
    Convert a FeatureExtractionResult object into a flat dictionary.
    This makes it easier to display in tables and export to CSV/JSON.
    """
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

    # Add formants (F1, F2, F3, etc.) if present
    formants = timbre.get("formants", {})
    for key, value in formants.items():
        flat[key] = safe_round(value)

    # Add MFCC coefficients if present
    mfccs = timbre.get("mfccs", [])
    for idx, value in enumerate(mfccs, start=1):
        flat[f"mfcc_{idx}"] = safe_round(value)

    return flat


def display_overview(result) -> None:
    """Display key voice metrics in a dashboard layout with 4+3 columns."""
    rq = result.recording_quality
    pitch = result.features.get("pitch", {})
    voice_quality = result.features.get("voice_quality", {})

    # First row: recording quality and basic pitch
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

    # Second row: voice quality metrics
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

    # Disclaimer - these are research metrics, not a clinical diagnosis
    st.info(
        "These values describe recording quality, pitch behavior, and voice stability. "
        "They are useful for analysis, but they are not a medical diagnosis on their own."
    )


def display_pitch(result) -> None:
    """Display pitch-related features (mean, min, max, variability) in a table."""
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
    """Display formant frequencies (F1, F2, etc.) as a table and bar chart."""
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
    """Display MFCC coefficients as a table and line chart."""
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
    """Display recording quality and voice quality side by side."""
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
    """Store analysis result in session state history for later display."""
    flat = flatten_result(result)
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, flat)


def show_history() -> None:
    """Display and provide download for analysis history."""
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


def make_dialogue_turn_dataframe(result, source_filename: str) -> pd.DataFrame:
    """Convert dialogue diarization result to a DataFrame for editing."""
    processor = DialogueProcessor(VoiceFeatureExtractor(DEFAULT_CONFIG.model_copy(deep=True)))
    rows = processor.turns_to_manifest_rows(result.turns, source_filename=source_filename)
    df = pd.DataFrame(rows)
    return df[
        [
            "filename",
            "segment_id",
            "start_sec",
            "end_sec",
            "analysis_start_sec",
            "analysis_end_sec",
            "role",
            "speaker_id",
            "speaker_cluster",
            "turn_type",
            "is_barge_in",
            "gap_before_seconds",
            "gap_after_seconds",
            "text",
            "source",
        ]
    ]


def build_segments_from_manifest(df: pd.DataFrame) -> List[DialogueSegmentSpec]:
    """Parse user-edited manifest DataFrame back into DialogueSegmentSpec objects."""
    segments: List[DialogueSegmentSpec] = []
    for _, row in df.iterrows():
        role = row.get("role", "")
        if pd.isna(role):
            role = ""
        speaker_id = row.get("speaker_id", "")
        if pd.isna(speaker_id):
            speaker_id = ""
        text = row.get("text", "")
        if pd.isna(text):
            text = ""
        is_barge_in_value = row.get("is_barge_in")
        if pd.isna(is_barge_in_value):
            is_barge_in = None
        else:
            is_barge_in = str(is_barge_in_value).strip().lower() in {"1", "true", "yes", "y"}
        segments.append(
            DialogueSegmentSpec(
                segment_id=str(row.get("segment_id") or ""),
                start_sec=float(row.get("start_sec")),
                end_sec=float(row.get("end_sec")),
                analysis_start_sec=float(row.get("analysis_start_sec")) if pd.notna(
                    row.get("analysis_start_sec")) else None,
                analysis_end_sec=float(row.get("analysis_end_sec")) if pd.notna(row.get("analysis_end_sec")) else None,
                role=str(role).strip() or None,
                speaker_id=str(speaker_id).strip() or None,
                text=str(text).strip() or None,
                turn_type=str(row.get("turn_type")).strip() if pd.notna(row.get("turn_type")) else None,
                is_barge_in=is_barge_in,
                source=str(row.get("source") or "manifest"),
            )
        )
    return segments


def dialogue_role_options(max_assistants: int = 20) -> List[str]:
    """Generate allowed role names for the dialogue manifest editor."""
    roles = ["assistant_1", "user", "music"]
    roles.extend(f"assistant_{index}" for index in range(2, max_assistants + 1))
    return roles


def render_single_recording_tab() -> None:
    """
    Single recording workflow:
    1. User uploads an audio file
    2. System extracts acoustic features
    3. System predicts genotype (rs11046212 / ABCC9)
    4. Displays detailed feature visualizations
    """

    # Sidebar configuration controls
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

    # File upload widget
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"], key="single_upload")

    if uploaded_file is not None:
        st.audio(uploaded_file)

    analyze = st.button("Analyze voice", type="primary", use_container_width=True, key="single_analyze")

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

                extractor = VoiceFeatureExtractor(config)

                session_id = str(uuid.uuid4())
                with st.spinner("Processing audio and extracting features..."):
                    result = extractor.extract_features(audio_bytes, session_id, original_filename=uploaded_file.name)

                # Validate minimum duration
                actual_duration = result.recording_quality.get("duration_seconds", 0.0)
                if actual_duration < min_duration_seconds:
                    st.warning(
                        f"Audio is too short. Minimum duration is {min_duration_seconds}s, "
                        f"but the uploaded file is {safe_round(actual_duration)}s."
                    )
                else:
                    st.success("Feature extraction completed successfully.")

                # --- GENETIC PREDICTION SECTION ---
                st.subheader("Genetic Prediction (rs11046212 / ABCC9)")

                # Extract features needed for the genetic model
                pitch = result.features.get("pitch", {})
                voice_quality = result.features.get("voice_quality", {})

                genetic_features = {
                    'pitch_mean': pitch.get('mean_f0_hz', 0),
                    'pitch_variability': pitch.get('variability', 0),
                    'jitter': voice_quality.get('jitter_percent', 0) or 0,
                    'shimmer': voice_quality.get('shimmer_db', 0) or 0,
                    'hnr': voice_quality.get('harmonic_to_noise_ratio', 0) or 0,
                }

                try:
                    # Get genotype prediction from the ML model
                    prediction = predictor.predict(genetic_features)

                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        genotype = prediction['genotype']
                        # Display with appropriate color coding based on genotype
                        if genotype == "CC":
                            st.success(f"**Predicted Genotype: {genotype}** (Reference)")
                        elif genotype == "CT":
                            st.warning(f"**Predicted Genotype: {genotype}** (Heterozygous)")
                        else:
                            st.error(f"**Predicted Genotype: {genotype}** (Variant)")

                        st.caption(f"SNP: {prediction['snp']} | Gene: {prediction['gene']}")
                        st.info(prediction['clinical_note'])

                    with col_g2:
                        st.write("**Probabilities:**")
                        probs = prediction['probabilities']
                        # Show confidence bars for each possible genotype
                        st.progress(probs['CC'], text=f"CC: {probs['CC']:.1%}")
                        st.progress(probs['CT'], text=f"CT: {probs['CT']:.1%}")
                        st.progress(probs['TT'], text=f"TT: {probs['TT']:.1%}")

                except Exception as e:
                    st.warning(f"Genetic prediction unavailable: {e}")

                st.divider()

                # --- FEATURE VISUALIZATION SECTION ---
                st.subheader("Overview")
                display_overview(result)

                # Detailed feature tabs
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

                # Save to history and provide download
                add_result_to_history(result)

                json_payload = {
                    "session_id": result.session_id,
                    "recording_quality": result.recording_quality,
                    "features": result.features,
                    "processing_timestamp": result.processing_timestamp,
                }
                st.download_button(
                    label="Download result as JSON",
                    data=json.dumps(
                        json_payload,
                        indent=2,
                        default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x)
                    ),
                    file_name=f"voice_features_{result.session_id}.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Feature extraction failed: {e}")


def render_dialogue_tab() -> None:
    """
    Conversation dialogue workflow:
    1. User uploads a conversation audio (phone call, interview, etc.)
    2. System runs diarization to split into turns and identify speakers
    3. User can edit role labels (assistant_1, user, assistant_2, music)
    4. System extracts features only for user-labeled segments
    5. Predicts genotype based on user voice only
    """

    st.subheader("Conversation analysis")
    st.caption(
        "Use this flow for phone conversations. The first step is diarization; "
        "roles are assigned by matching each turn against stable voice profiles. "
        "assistant_1 is seeded from the first assistant voice, user from the first user voice, and new assistant voices become assistant_2, assistant_3, and so on. "
        "Hold music is marked as music only when the segment is strongly music-like. Barge-in turns can be trimmed before feature extraction."
    )

    # Dialogue-specific settings in sidebar
    with st.sidebar:
        st.header("Dialogue settings")
        dialogue_vad_top_db = st.slider("VAD threshold (top_db)", min_value=10.0, max_value=60.0, value=28.0, step=1.0)
        dialogue_min_turn = st.number_input("Min turn duration (s)", min_value=0.1, max_value=5.0, value=0.35,
                                            step=0.05)
        dialogue_merge_gap = st.number_input("Merge gap (s)", min_value=0.0, max_value=2.0, value=0.25, step=0.05)
        dialogue_target_sample_rate = st.selectbox(
            "Target sample rate (dialogue)", options=[8000, 16000, 22050, 44100], index=1, key="dialogue_sr"
        )
        st.markdown("---")
        st.caption("The manifest you download can be edited before user-only analysis.")

    # File upload for conversation audio
    uploaded_file = st.file_uploader(
        "Upload conversation audio",
        type=["wav", "mp3", "m4a"],
        key="dialogue_upload",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        diarize_clicked = st.button("Run diarization", type="primary", use_container_width=True, key="dialogue_diarize")
    with col_b:
        clear_clicked = st.button("Clear dialogue state", use_container_width=True, key="dialogue_clear")

    # Clear all dialogue-related session state
    if clear_clicked:
        st.session_state.pop("dialogue_diarization", None)
        st.session_state.pop("dialogue_manifest_df", None)
        st.session_state.pop("dialogue_analysis", None)
        st.rerun()

    # Run diarization on uploaded conversation
    if diarize_clicked:
        if uploaded_file is None:
            st.error("Please upload a conversation audio file first.")
        else:
            try:
                audio_bytes = uploaded_file.read()
                if len(audio_bytes) > 50 * 1024 * 1024:
                    st.error("File too large. Maximum size is 50MB.")
                    st.stop()

                config = make_config(
                    pitch_min_f0=75.0,
                    pitch_max_f0=300.0,
                    formant_max_frequency=5500.0,
                    formant_number=4,
                    mfcc_number=13,
                    min_duration_seconds=0.5,
                    target_sample_rate=dialogue_target_sample_rate,
                    vad_top_db=dialogue_vad_top_db,
                    min_turn_duration_seconds=dialogue_min_turn,
                    merge_gap_seconds=dialogue_merge_gap,
                )
                processor = DialogueProcessor(VoiceFeatureExtractor(config))
                session_id = str(uuid.uuid4())
                with st.spinner("Running diarization on the conversation..."):
                    result = processor.diarize_dialogue(audio_bytes, session_id, original_filename=uploaded_file.name)

                # Store results in session state for persistence across reruns
                st.session_state.dialogue_diarization = {
                    "source_filename": uploaded_file.name,
                    "result": result,
                    "audio_bytes": audio_bytes,
                    "config": config,
                }
                st.session_state.dialogue_manifest_df = make_dialogue_turn_dataframe(result, uploaded_file.name)
                st.session_state.pop("dialogue_analysis", None)
                st.success("Diarization completed.")
            except Exception as e:
                st.error(f"Diarization failed: {e}")

    # Check if we have diarization results
    state = st.session_state.get("dialogue_diarization")
    if not state:
        st.info("Upload a conversation and run diarization to generate a dialogue manifest.")
        return

    result = state["result"]
    source_filename = state["source_filename"]

    # Display diarization summary
    st.subheader("Diarization output")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Speech turns", result.diarization.get("speech_turn_count", 0))
    with c2:
        st.metric("Speaker clusters", result.diarization.get("speaker_cluster_count", 0))
    with c3:
        st.metric("Source file", source_filename)

    st.caption(
        "Role labels are semantic: the first turn seeds assistant_1, the first user turn seeds user, later distinct assistant voices become assistant_2, assistant_3, and so on, and hold music is marked as music. "
        "speaker_id is a stable canonical voice label like speaker_1, while speaker_cluster is the technical diarization cluster."
    )
    st.dataframe(pd.DataFrame(result.turns), use_container_width=True, hide_index=True)

    # Editable manifest for role assignment
    st.subheader("Generate dialogue_manifest.csv")
    manifest_df = st.data_editor(
        st.session_state.dialogue_manifest_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "role": st.column_config.SelectboxColumn(
                "role",
                options=dialogue_role_options(),
                required=True,
            ),
            "speaker_id": st.column_config.TextColumn("speaker_id"),
            "text": st.column_config.TextColumn("text"),
        },
        disabled=[
            "filename",
            "segment_id",
            "start_sec",
            "end_sec",
            "analysis_start_sec",
            "analysis_end_sec",
            "speaker_cluster",
            "turn_type",
            "is_barge_in",
            "gap_before_seconds",
            "gap_after_seconds",
            "source",
        ],
        key="dialogue_manifest_editor",
    )
    st.session_state.dialogue_manifest_df = manifest_df

    # Download the manifest as CSV
    manifest_csv = manifest_df.to_csv(index=False)
    st.download_button(
        label="Download dialogue_manifest.csv",
        data=manifest_csv,
        file_name="dialogue_manifest.csv",
        mime="text/csv",
    )

    # Role-aware analysis - extracts features only for user-labeled segments
    st.subheader("Role-aware analysis")
    st.caption("The analysis below uses only rows labeled as role=user.")

    analyze_labeled = st.button(
        "Analyze labeled dialogue",
        type="primary",
        use_container_width=True,
        key="dialogue_analyze_labeled",
    )

    if analyze_labeled:
        try:
            segments = build_segments_from_manifest(manifest_df)
            processor = DialogueProcessor(VoiceFeatureExtractor(state["config"]))
            analysis = processor.analyze_labeled_dialogue(
                state["audio_bytes"],
                str(uuid.uuid4()),
                segments,
                original_filename=source_filename,
                strict_roles=True,
            )
            st.session_state.dialogue_analysis = analysis
            st.success("Role-aware analysis completed.")
        except Exception as e:
            st.error(f"Role-aware analysis failed: {e}")

    # Display analysis results if available
    analysis = st.session_state.get("dialogue_analysis")
    if analysis:
        st.metric("User turns", analysis.diarization.get("user_turn_count", 0))
        user_summary_df = pd.DataFrame(
            [
                {"feature": key, "value": value}
                for key, value in analysis.user_summary["feature_map"].items()
            ]
        )
        st.dataframe(user_summary_df, use_container_width=True, hide_index=True)

        # --- GENETIC PREDICTION FOR USER VOICE ---
        st.subheader("Genetic Prediction for User Voice")

        feature_map = analysis.user_summary.get("feature_map", {})

        # Extract average features from all user turns
        genetic_features = {
            'pitch_mean': feature_map.get('mean_f0_hz_mean', 0),
            'pitch_variability': feature_map.get('variability_mean', 0),
            'jitter': feature_map.get('jitter_percent_mean', 0) or 0,
            'shimmer': feature_map.get('shimmer_db_mean', 0) or 0,
            'hnr': feature_map.get('harmonic_to_noise_ratio_mean', 0) or 0,
        }

        try:
            prediction = predictor.predict(genetic_features)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                genotype = prediction['genotype']
                if genotype == "CC":
                    st.success(f"**Predicted Genotype: {genotype}** (Reference)")
                elif genotype == "CT":
                    st.warning(f"**Predicted Genotype: {genotype}** (Heterozygous)")
                else:
                    st.error(f"**Predicted Genotype: {genotype}** (Variant)")

                st.caption(f"SNP: {prediction['snp']} | Gene: {prediction['gene']}")
                st.info(prediction['clinical_note'])

            with col_g2:
                st.write("**Probabilities:**")
                probs = prediction['probabilities']
                st.progress(probs['CC'], text=f"CC: {probs['CC']:.1%}")
                st.progress(probs['CT'], text=f"CT: {probs['CT']:.1%}")
                st.progress(probs['TT'], text=f"TT: {probs['TT']:.1%}")

        except Exception as e:
            st.warning(f"Genetic prediction unavailable: {e}")

        # Download button for full analysis results
        st.download_button(
            label="Download user feature summary as JSON",
            data=json.dumps(
                {
                    "session_id": analysis.session_id,
                    "source_filename": analysis.source_filename,
                    "recording_quality": analysis.recording_quality,
                    "diarization": analysis.diarization,
                    "speaker_summary": analysis.speaker_summary,
                    "user_summary": analysis.user_summary,
                    "turns": analysis.turns,
                    "processing_timestamp": analysis.processing_timestamp,
                },
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x)
            ),
            file_name="dialogue_user_features.json",
            mime="application/json",
        )


def main() -> None:
    """
    Main entry point for the Streamlit application.
    Sets up the UI layout with two main tabs: Single recording and Conversation recording.
    """
    st.title("Voice Genetics")
    st.markdown(
        "Analyze either a single voice recording or a conversation. "
        "The dialogue flow generates a `dialogue_manifest.csv` and extracts user-only features."
    )

    # Educational expander explaining all metrics
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

    # Main tab layout
    single_tab, dialogue_tab = st.tabs(["Single recording", "Conversation recording"])
    with single_tab:
        render_single_recording_tab()
    with dialogue_tab:
        render_dialogue_tab()

    st.markdown("---")
    show_history()

    # Help section with run instructions
    with st.expander("How to run this app"):
        st.code(
            "pip install -r requirements.txt\n"
            "python -m streamlit run streamlit_app.py",
            language="bash",
        )


if __name__ == "__main__":
    main()