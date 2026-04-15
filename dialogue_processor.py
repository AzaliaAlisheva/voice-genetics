from __future__ import annotations

import datetime
import csv
import io
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import librosa
import parselmouth
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from audio_processor import FeatureExtractionResult, VoiceFeatureExtractor

# Standard role labels used throughout the system
ASSISTANT_1_ROLE = "assistant_1"  # Primary assistant/agent voice
USER_ROLE = "user"  # Customer/client/user voice
ASSISTANT_2_ROLE = "assistant_2"  # Secondary assistant voice (if present)
MUSIC_ROLE = "music"  # Hold music or non-speech audio

LOGGER = logging.getLogger(__name__)

# Aliases for role normalization - maps various input strings to standard roles
ROLE_ALIASES = {
    "assistant": ASSISTANT_1_ROLE,
    "agent": ASSISTANT_1_ROLE,
    "assistant_1": ASSISTANT_1_ROLE,
    "assistant_voice": ASSISTANT_1_ROLE,
    "assistantvoice": ASSISTANT_1_ROLE,
    "caller": USER_ROLE,
    "client": USER_ROLE,
    "customer": USER_ROLE,
    "human": USER_ROLE,
    "user": USER_ROLE,
    "music": MUSIC_ROLE,
    "hold_music": MUSIC_ROLE,
    "holdmusic": MUSIC_ROLE,
    "hold": MUSIC_ROLE,
    "jingle": MUSIC_ROLE,
}


def _normalize_role(value: Optional[str]) -> str:
    """
    Convert various role string representations to a canonical role label.
    """
    if not value:
        return "unknown"
    role = value.strip().lower().replace("-", "_").replace(" ", "_")
    if role in ROLE_ALIASES:
        return ROLE_ALIASES[role]
    if role.startswith("assistant_"):
        suffix = role.removeprefix("assistant_")
        if suffix.isdigit():
            return role
    if role == "assistant":
        return ASSISTANT_1_ROLE
    if role == "music":
        return MUSIC_ROLE
    if role == "user":
        return USER_ROLE
    return role


def _is_known_role(role: str) -> bool:
    """Check if a role string is a valid, known role label."""
    normalized = _normalize_role(role)
    if normalized in {USER_ROLE, MUSIC_ROLE, ASSISTANT_1_ROLE}:
        return True
    if normalized.startswith("assistant_"):
        suffix = normalized.removeprefix("assistant_")
        return suffix.isdigit()
    return False


def _assistant_role_for_index(index: int) -> str:
    """Generate assistant role name like 'assistant_2', 'assistant_3', etc."""
    return f"assistant_{index}"


def _speaker_label(index: Optional[int]) -> Optional[str]:
    """Generate stable speaker ID like 'speaker_1', 'speaker_2', etc."""
    if index is None:
        return None
    return f"speaker_{index}"


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Used for comparing voice embeddings to match speakers.
    """
    a = np.asarray(vec_a, dtype=float).ravel()
    b = np.asarray(vec_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class DialogueSegmentSpec:
    """
    Specification for a single dialogue segment to analyze.

    This is the input format for the analyze_labeled_dialogue method.
    Users can edit the manifest CSV and this class parses it back.
    """
    segment_id: str  # Unique identifier for this segment
    start_sec: float  # Start time in seconds
    end_sec: float  # End time in seconds
    analysis_start_sec: Optional[float] = None  # Trimmed start (for barge-in removal)
    analysis_end_sec: Optional[float] = None  # Trimmed end
    role: Optional[str] = None  # assistant_1, user, assistant_2, music
    speaker_id: Optional[str] = None  # Stable speaker identifier
    text: Optional[str] = None  # Optional transcript
    turn_type: Optional[str] = None  # 'speech', 'music', 'barge_in'
    is_barge_in: Optional[bool] = None  # Whether this turn interrupts previous
    source: str = "manifest"  # Source of this segment data


@dataclass
class DialogueTurnResult:
    """Result of analyzing a single dialogue turn."""
    turn_id: str
    segment_id: Optional[str]
    start_sec: float
    end_sec: float
    duration_seconds: float
    role: str
    speaker_id: Optional[str]
    speaker_cluster: Optional[int]
    source: str
    text: Optional[str]
    recording_quality: Dict[str, Any]
    features: Dict[str, Any]


@dataclass
class DialogueAnalysisResult:
    """Complete analysis result for a conversation."""
    session_id: str
    source_filename: Optional[str]
    recording_quality: Dict[str, Any]
    diarization: Dict[str, Any]  # Speaker clustering and role info
    speaker_summary: Dict[str, Any]  # Aggregated stats by role/cluster
    user_summary: Dict[str, Any]  # Aggregated features for user role only
    turns: List[Dict[str, Any]]  # Individual turn data
    processing_timestamp: str


class DialogueProcessor:
    """
    Conversation-level processing with diarization and role-aware feature extraction.

    This class handles the complete pipeline for analyzing multi-speaker conversations:
    1. Turn detection (energy-based VAD)
    2. Speaker clustering (agglomerative clustering on voice embeddings)
    3. Role assignment (assistant_1, user, assistant_N, music)
    4. Feature extraction (per turn)
    5. Aggregation (user-only summary for genetic prediction)
    """

    def __init__(self, extractor: VoiceFeatureExtractor):
        """Initialize with a configured VoiceFeatureExtractor instance."""
        self.extractor = extractor

    def _waveform_from_bytes(self, audio_data: bytes, original_filename: Optional[str] = None):
        """Load audio from bytes and return waveform, sample rate, and Sound object."""
        return self.extractor.load_audio(audio_data, original_filename=original_filename)

    def _turn_embedding_dim(self) -> int:
        """Calculate the dimension of the embedding vector for a speech turn."""
        mfcc_count = int(self.extractor.config.mfcc_number or 12)
        # MFCC mean + MFCC std + spectral features (10) + pitch features (4)
        return mfcc_count * 2 + 10 + 4

    def _validate_time_range(
            self,
            start_sec: float,
            end_sec: float,
            *,
            audio_duration_seconds: float,
            label: str,
            min_sec: float = 0.0,
            max_sec: Optional[float] = None,
    ) -> None:
        """
        Validate that a time range is within audio bounds.
        """
        if not np.isfinite(start_sec) or not np.isfinite(end_sec):
            raise ValueError(f"{label} has non-finite time bounds")
        if start_sec < min_sec:
            raise ValueError(f"{label} starts before {min_sec:.3f}s")
        if end_sec <= start_sec:
            raise ValueError(f"{label} end_sec must be greater than start_sec")
        upper_bound = audio_duration_seconds if max_sec is None else min(audio_duration_seconds, max_sec)
        if end_sec > upper_bound + 1e-6:
            raise ValueError(f"{label} ends after the audio boundary ({upper_bound:.3f}s)")

    def turns_to_manifest_rows(
            self,
            turns: Sequence[Dict[str, Any]],
            *,
            source_filename: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert diarization turns to rows ready for CSV manifest.

        This creates a DataFrame-friendly structure that users can edit
        to correct role assignments before re-analysis.
        """
        rows: List[Dict[str, Any]] = []
        for turn in turns:
            cluster_id = turn.get("speaker_cluster")
            speaker_id = turn.get("speaker_id")
            if not speaker_id and cluster_id is not None:
                speaker_id = f"speaker_{cluster_id}"

            role = _normalize_role(turn.get("role"))

            rows.append(
                {
                    "filename": source_filename or "",
                    "segment_id": turn.get("segment_id") or turn.get("turn_id") or "",
                    "start_sec": turn.get("start_sec"),
                    "end_sec": turn.get("end_sec"),
                    "analysis_start_sec": turn.get("analysis_start_sec", turn.get("start_sec")),
                    "analysis_end_sec": turn.get("analysis_end_sec", turn.get("end_sec")),
                    "role": role,
                    "speaker_id": speaker_id or "",
                    "speaker_cluster": cluster_id if cluster_id is not None else "",
                    "turn_type": turn.get("turn_type") or "speech",
                    "is_barge_in": "true" if bool(turn.get("is_barge_in", False)) else "false",
                    "gap_before_seconds": turn.get("gap_before_seconds", 0.0),
                    "gap_after_seconds": turn.get("gap_after_seconds", 0.0),
                    "text": turn.get("text") or "",
                    "source": turn.get("source") or "energy_vad",
                }
            )
        return rows

    def turns_to_manifest_csv(
            self,
            turns: Sequence[Dict[str, Any]],
            *,
            source_filename: Optional[str] = None,
    ) -> str:
        """Generate CSV string from diarization turns for download."""
        rows = self.turns_to_manifest_rows(
            turns,
            source_filename=source_filename,
        )
        fieldnames = [
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
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    def _turn_embedding(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Build a compact embedding vector for a speech turn.

        The embedding combines:
        - MFCCs (mean and std) for spectral characteristics
        - Spectral features (centroid, bandwidth, rolloff, ZCR, RMS)
        - Pitch features (mean, min, max, variability)

        These embeddings are used for speaker clustering via cosine similarity.
        """
        mfcc_count = int(self.extractor.config.mfcc_number or 12)
        embedding_dim = self._turn_embedding_dim()
        if len(y) == 0:
            return np.zeros(embedding_dim, dtype=float)

        feature_blocks: List[float] = []

        def extend_stats(values: np.ndarray) -> None:
            """Add mean and std of a 1D array to feature blocks."""
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size == 0:
                feature_blocks.extend([0.0, 0.0])
            else:
                feature_blocks.extend([float(np.mean(arr)), float(np.std(arr))])

        def extend_vector(values: np.ndarray) -> None:
            """Add all values of a 1D array to feature blocks."""
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size == 0:
                feature_blocks.extend([0.0] * mfcc_count)
            else:
                feature_blocks.extend(arr.tolist())

        # Extract MFCC features (spectral envelope)
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_count)
            extend_vector(mfcc.mean(axis=1))
            extend_vector(mfcc.std(axis=1))
        except Exception as exc:
            LOGGER.debug("MFCC extraction failed for turn embedding: %s", exc, exc_info=True)
            feature_blocks.extend([0.0] * (mfcc_count * 2))

        # Extract spectral features (timbre characteristics)
        try:
            extend_stats(librosa.feature.spectral_centroid(y=y, sr=sr))
            extend_stats(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            extend_stats(librosa.feature.spectral_rolloff(y=y, sr=sr))
            extend_stats(librosa.feature.zero_crossing_rate(y))
            extend_stats(librosa.feature.rms(y=y))
        except Exception as exc:
            LOGGER.debug("Spectral feature extraction failed for turn embedding: %s", exc, exc_info=True)
            feature_blocks.extend([0.0] * 10)

        # Extract pitch features (fundamental frequency)
        try:
            pitch_features = self.extractor.extract_pitch_features(parselmouth.Sound(y, sr))
        except Exception as exc:
            LOGGER.debug("Pitch extraction failed for turn embedding: %s", exc, exc_info=True)
            pitch_features = None

        if pitch_features is not None:
            feature_blocks.extend(
                [
                    float(pitch_features.mean_f0_hz),
                    float(pitch_features.min_f0_hz),
                    float(pitch_features.max_f0_hz),
                    float(pitch_features.variability),
                ]
            )
        else:
            feature_blocks.extend([0.0, 0.0, 0.0, 0.0])

        return np.asarray(feature_blocks, dtype=float)

    def _turn_embeddings(
            self,
            y: np.ndarray,
            sr: int,
            turns: Sequence[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """Generate embeddings for all turns in a conversation."""
        embeddings: List[np.ndarray] = []
        for turn in turns:
            turn_audio = self._segment_waveform(y, sr, turn["start_sec"], turn["end_sec"])
            if len(turn_audio) == 0:
                embeddings.append(np.zeros(self._turn_embedding_dim(), dtype=float))
                continue
            embeddings.append(self._turn_embedding(turn_audio, sr))
        return embeddings

    def _segment_waveform(self, y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
        """Extract a time-bounded slice of the waveform."""
        return self.extractor.slice_waveform(y, sr, start_sec, end_sec)

    def _flatten_feature_result(self, result: FeatureExtractionResult) -> Dict[str, Any]:
        """Convert FeatureExtractionResult to a flat dictionary for aggregation."""
        recording_quality = result.recording_quality
        features = result.features
        pitch = features.get("pitch", {})
        timbre = features.get("timbre", {})
        voice_quality = features.get("voice_quality", {})

        flat: Dict[str, Any] = {
            "session_id": result.session_id,
            "segment_id": result.segment_id,
            "role": result.role or "unknown",
            "speaker_id": result.speaker_id,
            "start_sec": result.start_sec,
            "end_sec": result.end_sec,
            "duration_seconds": recording_quality.get("duration_seconds"),
            "snr_db": recording_quality.get("snr_db"),
            "background_noise_level": recording_quality.get("background_noise_level"),
            "sample_rate": recording_quality.get("sample_rate"),
            "mean_f0_hz": pitch.get("mean_f0_hz"),
            "min_f0_hz": pitch.get("min_f0_hz"),
            "max_f0_hz": pitch.get("max_f0_hz"),
            "variability": pitch.get("variability"),
            "jitter_percent": voice_quality.get("jitter_percent"),
            "shimmer_db": voice_quality.get("shimmer_db"),
            "harmonic_to_noise_ratio": voice_quality.get("harmonic_to_noise_ratio"),
        }

        # Add formants (F1, F2, etc.)
        for key, value in timbre.get("formants", {}).items():
            flat[key] = value

        # Add MFCC coefficients
        for idx, value in enumerate(timbre.get("mfccs", []), start=1):
            flat[f"mfcc_{idx}"] = value

        return flat

    def _aggregate_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate numeric features across multiple rows.

        Computes mean, std, min, max, and count for each numeric field.
        This is used to create summary statistics for user segments.
        """
        numeric_buckets: Dict[str, List[float]] = defaultdict(list)

        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    numeric_buckets[key].append(float(value))

        aggregates: Dict[str, Any] = {}
        for key, values in numeric_buckets.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            aggregates[f"{key}_mean"] = float(np.mean(arr))
            aggregates[f"{key}_std"] = float(np.std(arr))
            aggregates[f"{key}_min"] = float(np.min(arr))
            aggregates[f"{key}_max"] = float(np.max(arr))
            aggregates[f"{key}_count"] = int(arr.size)

        ordered_keys = sorted(aggregates)
        return {
            "feature_names": ordered_keys,
            "feature_vector": [aggregates[key] for key in ordered_keys],
            "feature_map": aggregates,
        }

    def _speaker_summary(self, turn_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics grouped by role, cluster, and speaker identity.

        Returns:
            Dictionary with 'by_role', 'by_cluster', and 'by_identity' summaries
            each containing segment count and duration statistics.
        """
        by_role: Dict[str, Dict[str, Any]] = {}
        by_cluster: Dict[str, Dict[str, Any]] = {}
        by_identity: Dict[str, Dict[str, Any]] = {}

        role_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        cluster_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        identity_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in turn_rows:
            role_buckets[_normalize_role(row.get("role"))].append(row)
            cluster_key = row.get("speaker_cluster")
            cluster_label = str(cluster_key) if cluster_key is not None else "unknown"
            cluster_buckets[cluster_label].append(row)

            identity_key = cluster_key
            if identity_key is None:
                identity_key = row.get("speaker_id")
            identity_label = str(identity_key) if identity_key is not None else "unknown"
            identity_buckets[identity_label].append(row)

        for role, items in role_buckets.items():
            durations = np.asarray([float(item.get("duration_seconds", 0.0)) for item in items], dtype=float)
            by_role[role] = {
                "segment_count": len(items),
                "total_duration_seconds": float(durations.sum()) if durations.size else 0.0,
                "mean_duration_seconds": float(durations.mean()) if durations.size else 0.0,
            }

        for cluster_id, items in cluster_buckets.items():
            durations = np.asarray([float(item.get("duration_seconds", 0.0)) for item in items], dtype=float)
            by_cluster[cluster_id] = {
                "segment_count": len(items),
                "total_duration_seconds": float(durations.sum()) if durations.size else 0.0,
                "mean_duration_seconds": float(durations.mean()) if durations.size else 0.0,
            }

        for identity_id, items in identity_buckets.items():
            durations = np.asarray([float(item.get("duration_seconds", 0.0)) for item in items], dtype=float)
            by_identity[identity_id] = {
                "segment_count": len(items),
                "total_duration_seconds": float(durations.sum()) if durations.size else 0.0,
                "mean_duration_seconds": float(durations.mean()) if durations.size else 0.0,
            }

        return {"by_role": by_role, "by_cluster": by_cluster, "by_identity": by_identity}

    def _detect_speech_turns(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect speech turns using energy-based Voice Activity Detection (VAD)."""
        intervals = librosa.effects.split(y, top_db=self.extractor.config.vad_top_db)
        if len(intervals) == 0:
            return []

        min_turn_samples = int(self.extractor.config.min_turn_duration_seconds * sr)
        merge_gap_samples = int(self.extractor.config.merge_gap_seconds * sr)

        # Merge intervals that are close together
        merged: List[List[int]] = []
        for start, end in intervals:
            if not merged:
                merged.append([int(start), int(end)])
                continue
            if int(start) - merged[-1][1] <= merge_gap_samples:
                merged[-1][1] = int(end)
            else:
                merged.append([int(start), int(end)])

        # Convert to turn objects, filtering short segments
        turns: List[Dict[str, Any]] = []
        for idx, (start, end) in enumerate(merged, start=1):
            if end - start < min_turn_samples:
                continue
            turns.append(
                {
                    "turn_id": f"turn_{idx:03d}",
                    "segment_id": f"turn_{idx:03d}",
                    "start_sec": start / sr,
                    "end_sec": end / sr,
                    "duration_seconds": (end - start) / sr,
                    "source": "energy_vad",
                }
            )

        return turns

    def _cluster_turns(self, y: np.ndarray, sr: int, turns: Sequence[Dict[str, Any]]) -> List[int]:
        """
        Cluster speech turns into speaker groups using agglomerative clustering.

        Returns a list of cluster labels (integers) for each turn.
        """
        if len(turns) <= 1:
            return [0] * len(turns)

        # Generate embeddings and normalize
        embeddings = self._turn_embeddings(y, sr, turns)
        matrix = np.vstack(embeddings)
        matrix = normalize(matrix, norm="l2")

        # Configure clustering parameters
        max_speakers = max(2, int(self.extractor.config.max_dialogue_speakers or len(turns)))
        threshold = float(self.extractor.config.dialogue_cluster_distance_threshold or 0.04)
        min_threshold = 0.01
        max_threshold = 0.15
        step = 0.01

        def fit_clusters(distance_threshold: float) -> np.ndarray:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage="average",
                metric="cosine",
            )
            return clustering.fit_predict(matrix)

        labels = fit_clusters(threshold)
        cluster_count = len(set(labels.tolist()))

        # Adjust threshold to get desired number of speakers
        if cluster_count > max_speakers:
            current_threshold = threshold
            while cluster_count > max_speakers and current_threshold < max_threshold:
                current_threshold = min(max_threshold, current_threshold + step)
                labels = fit_clusters(current_threshold)
                cluster_count = len(set(labels.tolist()))

        elif cluster_count == 1 and len(turns) > 2:
            current_threshold = threshold
            while cluster_count == 1 and current_threshold > min_threshold:
                current_threshold = max(min_threshold, current_threshold - step)
                labels = fit_clusters(current_threshold)
                cluster_count = len(set(labels.tolist()))

        return labels.tolist()

    def _is_music_like(
            self,
            y: np.ndarray,
            sr: int,
            *,
            gap_before_seconds: float = 0.0,
            gap_after_seconds: float = 0.0,
    ) -> bool:
        """
        Conservative heuristic to detect hold music / non-speech segments.

        Uses multiple features:
        - Spectral flatness (music is flatter than speech)
        - Beat tracking (music has regular beats)
        - Pitch variability (speech varies more than music)
        - Duration and gap context (music usually longer with gaps)
        """
        if len(y) == 0 or sr <= 0:
            return False

        duration_seconds = len(y) / sr
        min_duration = float(self.extractor.config.music_min_duration_seconds or 6.0)
        min_gap = float(self.extractor.config.music_min_gap_seconds or 0.75)
        min_score = int(self.extractor.config.music_min_score or 4)

        if duration_seconds < min_duration:
            return False
        if gap_before_seconds < min_gap and gap_after_seconds < min_gap:
            return False

        flatness_score = 1.0
        beat_count = 0
        beat_regularity = float("inf")
        tempo = 0.0
        pitch_variability = 1.0

        # Spectral flatness - music is typically flatter (more noise-like spectrum)
        try:
            flatness_score = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        except Exception as exc:
            LOGGER.debug("Spectral flatness failed for music heuristic: %s", exc, exc_info=True)
            flatness_score = 1.0

        # Beat tracking - music has regular, detectable beats
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            beat_count = len(beat_frames)
            if beat_count >= 4:
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                intervals = np.diff(beat_times)
                if intervals.size > 0 and np.mean(intervals) > 0:
                    beat_regularity = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
        except Exception as exc:
            LOGGER.debug("Beat tracking failed for music heuristic: %s", exc, exc_info=True)
            tempo = 0.0
            beat_count = 0
            beat_regularity = float("inf")

        # Pitch variability - music has less pitch variation than speech
        try:
            pitch_features = self.extractor.extract_pitch_features(parselmouth.Sound(y, sr))
            pitch_variability = float(pitch_features.variability or 0.0)
        except Exception as exc:
            LOGGER.debug("Pitch extraction failed for music heuristic: %s", exc, exc_info=True)
            pitch_variability = 1.0

        # Quick rejection for clear non-music
        if flatness_score > 0.18:
            return False
        if beat_count < 6:
            return False
        if beat_regularity >= 0.18:
            return False
        if not (70.0 <= float(tempo) <= 170.0):
            return False
        if pitch_variability > 0.12:
            return False

        # Score-based classification
        music_score = 0
        if flatness_score < 0.15:
            music_score += 1
        if beat_count >= 6 and beat_regularity < 0.18:
            music_score += 1
        if 70.0 <= float(tempo) <= 170.0 and beat_count >= 6:
            music_score += 1
        if pitch_variability <= 0.08:
            music_score += 1

        return music_score >= min_score

    def _assign_diarization_roles(self, y: np.ndarray, sr: int, turn_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assign semantic roles (assistant_1, user, assistant_N, music) to diarized turns.

        Role assignment logic:
        1. First turn seeds assistant_1 voice profile
        2. Second turn seeds user voice profile
        3. Subsequent turns matched against existing profiles via cosine similarity
        4. If similarity high enough, assign matching role
        5. If music-like and dissimilar, assign music role
        6. Otherwise, create new assistant_N role

        Returns mapping from clusters to roles and speaker IDs.
        """
        if not turn_rows:
            return {
                "assistant_cluster": None,
                "user_cluster": None,
                "assistant_seed_cluster": None,
                "user_seed_cluster": None,
                "assistant_clusters": [],
                "music_clusters": [],
                "distinct_speaker_count": 0,
                "role_assignment": "voice_similarity_matching_with_optional_music",
                "speaker_role_map": {},
                "speaker_cluster_to_id": {},
                "cluster_to_role": {},
                "role_to_cluster": {},
            }

        assistant_clusters: List[int] = []
        assistant_roles: List[str] = []
        music_clusters: List[int] = []
        cluster_to_speaker_ids: Dict[int, List[str]] = defaultdict(list)
        cluster_to_roles: Dict[int, List[str]] = defaultdict(list)
        speaker_role_votes: Dict[str, List[str]] = defaultdict(list)
        role_to_cluster: Dict[str, Optional[int]] = {}

        # Centroids for each role's voice profile
        role_centroids: Dict[str, np.ndarray] = {}
        role_counts: Dict[str, int] = defaultdict(int)
        role_to_speaker_id: Dict[str, str] = {}
        next_assistant_index = 2
        next_speaker_index = 1

        embeddings = self._turn_embeddings(y, sr, turn_rows)
        speaker_similarity_threshold = float(self.extractor.config.speaker_role_similarity_threshold or 0.78)
        music_similarity_threshold = float(self.extractor.config.music_role_similarity_threshold or 0.62)

        def get_role_speaker_id(role: str) -> str:
            """Generate stable speaker ID for a role (speaker_1, speaker_2, music_1)."""
            nonlocal next_speaker_index
            if role not in role_to_speaker_id:
                if role == MUSIC_ROLE:
                    role_to_speaker_id[role] = "music_1"
                else:
                    role_to_speaker_id[role] = _speaker_label(next_speaker_index) or "speaker_unknown"
                    next_speaker_index += 1
            return role_to_speaker_id[role]

        def update_centroid(role: str, embedding: np.ndarray) -> None:
            """Update running centroid for a role with new embedding."""
            if role not in role_centroids:
                role_centroids[role] = embedding.astype(float)
                role_counts[role] = 1
                return
            count = role_counts[role]
            role_centroids[role] = (role_centroids[role] * count + embedding) / float(count + 1)
            role_counts[role] = count + 1

        def majority_vote(values: Sequence[str]) -> Optional[str]:
            """Return most common value, breaking ties by first occurrence."""
            if not values:
                return None
            counts: Dict[str, int] = defaultdict(int)
            first_index: Dict[str, int] = {}
            for index, value in enumerate(values):
                counts[value] += 1
                first_index.setdefault(value, index)
            ordered = sorted(counts.items(), key=lambda item: (-item[1], first_index[item[0]]))
            return ordered[0][0] if ordered else None

        # Seed voice profiles from first two turns
        seed_roles = [ASSISTANT_1_ROLE]
        if len(turn_rows) > 1:
            seed_roles.append(USER_ROLE)

        for index, role in enumerate(seed_roles):
            update_centroid(role, embeddings[index])

        # Assign roles to all turns
        for index, row in enumerate(turn_rows):
            cluster_id_raw = row.get("speaker_cluster")
            cluster_id = int(cluster_id_raw) if cluster_id_raw is not None else None
            start_sec = float(row.get("start_sec", 0.0))
            end_sec = float(row.get("end_sec", start_sec))

            # Calculate gaps to previous/next turns
            gap_before = 0.0
            gap_after = 0.0
            if index > 0:
                prev_end = float(turn_rows[index - 1].get("end_sec", start_sec))
                gap_before = max(0.0, start_sec - prev_end)
            if index < len(turn_rows) - 1:
                next_start = float(turn_rows[index + 1].get("start_sec", end_sec))
                gap_after = max(0.0, next_start - end_sec)

            turn_audio = self._segment_waveform(y, sr, start_sec, end_sec)
            music_like = self._is_music_like(
                turn_audio,
                sr,
                gap_before_seconds=gap_before,
                gap_after_seconds=gap_after,
            )

            # Role assignment logic
            if index == 0:
                role = ASSISTANT_1_ROLE
            elif index == 1:
                role = USER_ROLE
            else:
                embedding = embeddings[index]
                best_role = None
                best_similarity = -1.0

                # Find closest matching existing role
                for candidate_role, centroid in role_centroids.items():
                    if centroid is None:
                        continue
                    similarity = _cosine_similarity(embedding, centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_role = candidate_role

                if music_like and best_similarity < music_similarity_threshold:
                    role = MUSIC_ROLE
                elif best_role is not None and best_similarity >= speaker_similarity_threshold:
                    role = best_role
                else:
                    role = _assistant_role_for_index(next_assistant_index)
                    next_assistant_index += 1

                update_centroid(role, embedding)

            # Update row with assigned role and metadata
            speaker_id = get_role_speaker_id(role)
            row["role"] = role
            row["speaker_id"] = speaker_id
            row["gap_before_seconds"] = gap_before
            row["gap_after_seconds"] = gap_after
            row["turn_type"] = "music" if role == MUSIC_ROLE else "speech"
            row["is_barge_in"] = index > 0 and row["role"] == USER_ROLE and gap_before <= float(
                self.extractor.config.barge_in_gap_seconds or 0.35)
            row["analysis_start_sec"] = (
                min(end_sec, start_sec + float(self.extractor.config.barge_in_trim_seconds or 0.0))
                if row["is_barge_in"]
                else start_sec
            )
            row["analysis_end_sec"] = end_sec

            # Track clusters by role
            if role.startswith("assistant_") and role not in assistant_roles:
                assistant_roles.append(role)
            if role.startswith("assistant_") and cluster_id is not None and cluster_id not in assistant_clusters:
                assistant_clusters.append(cluster_id)
            elif role == MUSIC_ROLE and cluster_id is not None and cluster_id not in music_clusters:
                music_clusters.append(cluster_id)

            speaker_role_votes[speaker_id].append(role)
            if cluster_id is not None:
                cluster_to_speaker_ids[cluster_id].append(speaker_id)
                cluster_to_roles[cluster_id].append(role)
            if cluster_id is not None:
                role_to_cluster.setdefault(role, cluster_id)

        # Post-process: resolve cluster to role mappings via majority vote
        distinct_speaker_count = len(
            {
                row.get("speaker_cluster")
                for row in turn_rows
                if row.get("speaker_cluster") is not None
            }
        )

        cluster_to_speaker_id: Dict[int, str] = {}
        cluster_to_role: Dict[int, str] = {}
        for cluster_id, speaker_ids in cluster_to_speaker_ids.items():
            chosen_speaker_id = majority_vote(speaker_ids)
            if chosen_speaker_id is not None:
                cluster_to_speaker_id[cluster_id] = chosen_speaker_id
        for cluster_id, roles in cluster_to_roles.items():
            chosen_role = majority_vote(roles)
            if chosen_role is not None:
                cluster_to_role[cluster_id] = chosen_role

        speaker_role_map: Dict[str, str] = {}
        for speaker_id, roles in speaker_role_votes.items():
            chosen_role = majority_vote(roles)
            if chosen_role is not None:
                speaker_role_map[speaker_id] = chosen_role

        assistant_seed_cluster = role_to_cluster.get(ASSISTANT_1_ROLE)
        user_seed_cluster = role_to_cluster.get(USER_ROLE)

        return {
            "assistant_cluster": assistant_seed_cluster,
            "user_cluster": user_seed_cluster,
            "assistant_seed_cluster": assistant_seed_cluster,
            "user_seed_cluster": user_seed_cluster,
            "assistant_2_cluster": role_to_cluster.get(ASSISTANT_2_ROLE),
            "assistant_clusters": assistant_clusters,
            "assistant_roles": assistant_roles,
            "music_clusters": music_clusters,
            "distinct_speaker_count": distinct_speaker_count,
            "role_assignment": "voice_similarity_matching_with_optional_music",
            "speaker_role_map": speaker_role_map,
            "speaker_cluster_to_id": cluster_to_speaker_id,
            "cluster_to_role": cluster_to_role,
            "role_to_cluster": role_to_cluster,
        }

    def diarize_dialogue(
            self,
            audio_data: bytes,
            session_id: str,
            *,
            original_filename: Optional[str] = None,
    ) -> DialogueAnalysisResult:
        """
        Main entry point for conversation diarization.

        Returns speech turns, speaker clusters, and semantic role labels.
        This is the first step in the dialogue analysis pipeline.
        """
        y, sr, _ = self._waveform_from_bytes(audio_data, original_filename=original_filename)
        recording_quality = self.extractor.extract_waveform_features(
            y,
            sr,
            session_id,
            source_filename=original_filename,
        ).recording_quality

        turns = self._detect_speech_turns(y, sr)
        cluster_ids = self._cluster_turns(y, sr, turns)

        turn_rows: List[Dict[str, Any]] = []
        for turn, cluster_id in zip(turns, cluster_ids):
            turn_rows.append(
                {
                    **turn,
                    "role": "unknown",
                    "speaker_id": None,
                    "speaker_cluster": int(cluster_id),
                    "analysis_start_sec": turn["start_sec"],
                    "analysis_end_sec": turn["end_sec"],
                    "gap_before_seconds": 0.0,
                    "gap_after_seconds": 0.0,
                    "turn_type": "speech",
                    "is_barge_in": False,
                    "recording_quality": {
                        "duration_seconds": turn["duration_seconds"],
                        "snr_db": None,
                        "background_noise_level": None,
                        "sample_rate": sr,
                    },
                    "features": {},
                }
            )

        role_assignment = self._assign_diarization_roles(y, sr, turn_rows)

        return DialogueAnalysisResult(
            session_id=session_id,
            source_filename=original_filename,
            recording_quality=recording_quality,
            diarization={
                "method": "energy_vad_plus_clustering",
                "role_assignment": role_assignment["role_assignment"],
                "speech_turn_count": len(turn_rows),
                "speaker_cluster_count": role_assignment["distinct_speaker_count"],
                "assistant_cluster": role_assignment["assistant_cluster"],
                "user_cluster": role_assignment["user_cluster"],
                "assistant_seed_cluster": role_assignment["assistant_seed_cluster"],
                "user_seed_cluster": role_assignment["user_seed_cluster"],
                "assistant_2_cluster": role_assignment["assistant_2_cluster"],
                "assistant_clusters": role_assignment["assistant_clusters"],
                "assistant_roles": role_assignment["assistant_roles"],
                "music_clusters": role_assignment["music_clusters"],
                "speaker_role_map": role_assignment["speaker_role_map"],
                "role_to_cluster": role_assignment["role_to_cluster"],
                "speaker_cluster_to_id": role_assignment["speaker_cluster_to_id"],
                "cluster_to_role": role_assignment["cluster_to_role"],
                "requires_role_annotation": False,
            },
            speaker_summary=self._speaker_summary(turn_rows),
            user_summary={},
            turns=turn_rows,
            processing_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def analyze_labeled_dialogue(
            self,
            audio_data: bytes,
            session_id: str,
            segments: Sequence[DialogueSegmentSpec],
            *,
            original_filename: Optional[str] = None,
            strict_roles: bool = True,
    ) -> DialogueAnalysisResult:
        """
        Extract features for labelled dialogue segments and aggregate user-only vectors.

        This is the second step after users edit the manifest.
        It extracts voice features only for segments marked as 'user'
        and aggregates them for genetic prediction.
        """
        y, sr, _ = self._waveform_from_bytes(audio_data, original_filename=original_filename)
        audio_duration_seconds = len(y) / sr if sr > 0 else 0.0
        recording_quality = self.extractor.extract_waveform_features(
            y,
            sr,
            session_id,
            source_filename=original_filename,
        ).recording_quality

        turn_rows: List[Dict[str, Any]] = []
        user_rows: List[Dict[str, Any]] = []

        for index, segment in enumerate(segments, start=1):
            role = _normalize_role(segment.role)
            if strict_roles and role == "unknown":
                raise ValueError(
                    f"Segment {segment.segment_id} has no role. "
                    "For dialogue analysis the manifest must label each segment with a semantic role."
                )
            if strict_roles and role != "unknown" and not _is_known_role(role):
                raise ValueError(
                    f"Segment {segment.segment_id} has unsupported role '{segment.role}'. "
                    "Allowed semantic roles include user, assistant, assistant_2+, and music."
                )

            analysis_start_sec = segment.analysis_start_sec if segment.analysis_start_sec is not None else segment.start_sec
            analysis_end_sec = segment.analysis_end_sec if segment.analysis_end_sec is not None else segment.end_sec
            self._validate_time_range(
                segment.start_sec,
                segment.end_sec,
                audio_duration_seconds=audio_duration_seconds,
                label=f"Segment {segment.segment_id}",
            )
            self._validate_time_range(
                analysis_start_sec,
                analysis_end_sec,
                audio_duration_seconds=audio_duration_seconds,
                label=f"Segment {segment.segment_id} analysis window",
                min_sec=segment.start_sec,
                max_sec=segment.end_sec,
            )
            segment_audio = self._segment_waveform(y, sr, analysis_start_sec, analysis_end_sec)
            if len(segment_audio) == 0:
                raise ValueError(
                    f"Segment {segment.segment_id} is empty after slicing. "
                    "Check start_sec and end_sec values."
                )

            segment_result = self.extractor.extract_waveform_features(
                segment_audio,
                sr,
                session_id,
                segment_id=segment.segment_id,
                role=role,
                speaker_id=segment.speaker_id,
                start_sec=analysis_start_sec,
                end_sec=analysis_end_sec,
                source_filename=original_filename,
            )

            flat = self._flatten_feature_result(segment_result)
            turn_rows.append(
                {
                    **asdict(segment),
                    "role": role,
                    "analysis_start_sec": analysis_start_sec,
                    "analysis_end_sec": analysis_end_sec,
                    "speaker_cluster": None,
                    "duration_seconds": flat["duration_seconds"],
                    "recording_quality": segment_result.recording_quality,
                    "features": segment_result.features,
                }
            )

            # Only collect user segments for genetic prediction
            if role == USER_ROLE:
                user_rows.append(flat)

        if not user_rows:
            raise ValueError(
                "No user segments were found. The manifest must mark the client voice as role=user."
            )

        user_summary = self._aggregate_rows(user_rows)
        speaker_summary = self._speaker_summary(turn_rows)

        return DialogueAnalysisResult(
            session_id=session_id,
            source_filename=original_filename,
            recording_quality=recording_quality,
            diarization={
                "method": "manifest_labeled_segments",
                "speech_turn_count": len(turn_rows),
                "user_turn_count": len(user_rows),
                "requires_role_annotation": False,
            },
            speaker_summary=speaker_summary,
            user_summary=user_summary,
            turns=turn_rows,
            processing_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )