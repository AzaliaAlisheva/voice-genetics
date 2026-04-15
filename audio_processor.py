import parselmouth
import numpy as np
import librosa
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RecordingQuality:
    duration_seconds: float
    snr_db: float
    background_noise_level: str
    sample_rate: int

@dataclass
class PitchFeatures:
    mean_f0_hz: float
    min_f0_hz: float
    max_f0_hz: float
    variability: float

@dataclass
class TimbreFeatures:
    mfccs: list
    formants: Dict[str, float]

@dataclass
class VoiceQuality:
    jitter_percent: Optional[float]
    shimmer_db: Optional[float]
    harmonic_to_noise_ratio: Optional[float] = None

@dataclass
class FeatureExtractionResult:
    session_id: str
    recording_quality: Dict[str, Any]
    features: Dict[str, Any]
    processing_timestamp: str
    segment_id: Optional[str] = None
    role: Optional[str] = None
    speaker_id: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    source_filename: Optional[str] = None

class VoiceFeatureExtractor:
    def __init__(self, config=None):
        self.config = config
        if self.config is None:
            from config import DEFAULT_CONFIG
            self.config = DEFAULT_CONFIG
    
    def load_audio(self, audio_data: bytes, original_filename: str = None) -> tuple:
        """Load audio from bytes and resample if needed."""
        try:
            suffix = ".wav"
            if original_filename:
                detected_suffix = Path(original_filename).suffix.lower()
                if detected_suffix:
                    suffix = detected_suffix

            tmp_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            # Load with librosa for resampling and feature extraction.
            y, sr = librosa.load(tmp_path, sr=self.config.target_sample_rate, mono=True)

            # Build Parselmouth sound from the decoded waveform so mp3/m4a inputs
            # work as long as librosa can decode them.
            sound = parselmouth.Sound(y, sr)

            return y, sr, sound
            
        except Exception as e:
            raise Exception(f"Error loading audio: {str(e)}")
        finally:
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def slice_waveform(self, y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
        """Return a time-bounded slice of the waveform."""
        if start_sec < 0:
            start_sec = 0.0
        if end_sec <= start_sec:
            raise ValueError("Segment end time must be greater than start time")

        start_sample = max(0, int(round(start_sec * sr)))
        end_sample = min(len(y), int(round(end_sec * sr)))
        if end_sample <= start_sample:
            raise ValueError("Segment bounds produce an empty slice")
        return y[start_sample:end_sample]
    
    def estimate_snr(self, y: np.ndarray, sr: int) -> float:
        """Estimate Signal-to-Noise Ratio"""
        # Simple noise estimation using median filtering
        noise_floor = np.median(np.abs(y)) * 1.5
        signal_power = np.mean(y**2)
        noise_power = noise_floor**2
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return min(60, max(0, snr))  # Clamp between 0 and 60 dB
        return 30.0  # Default if something goes wrong
    
    def assess_noise_level(self, snr_db: float) -> str:
        """Classify background noise level"""
        if snr_db >= 30:
            return "very_low"
        elif snr_db >= 20:
            return "low"
        elif snr_db >= 10:
            return "medium"
        else:
            return "high"
    
    def extract_pitch_features(self, sound: parselmouth.Sound) -> PitchFeatures:
        """Extract pitch-related features using Parselmouth"""
        try:
            pitch = sound.to_pitch(
                time_step=0.01,
                pitch_floor=self.config.pitch_min_f0,
                pitch_ceiling=self.config.pitch_max_f0
            )
            
            # Extract pitch values (excluding unvoiced frames)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            
            if len(pitch_values) == 0:
                return PitchFeatures(mean_f0_hz=0, min_f0_hz=0, max_f0_hz=0, variability=0)
            
            mean_f0 = np.mean(pitch_values)
            min_f0 = np.min(pitch_values)
            max_f0 = np.max(pitch_values)
            
            # Calculate variability (coefficient of variation)
            variability = np.std(pitch_values) / mean_f0 if mean_f0 > 0 else 0
            
            return PitchFeatures(
                mean_f0_hz=mean_f0,
                min_f0_hz=min_f0,
                max_f0_hz=max_f0,
                variability=float(variability)
            )
        except Exception as e:
            print(f"Error extracting pitch features: {e}")
            return PitchFeatures(mean_f0_hz=0, min_f0_hz=0, max_f0_hz=0, variability=0)
    
    def extract_formants(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """Extract formant frequencies using Parselmouth"""
        try:
            formant = sound.to_formant_burg(
                max_number_of_formants=self.config.formant_number,
                maximum_formant=self.config.formant_max_frequency
            )
            
            # Get formants at the midpoint of the sound
            time_point = sound.get_total_duration() / 2
            formants_dict = {}
            
            for i in range(1, self.config.formant_number + 1):
                f_value = formant.get_value_at_time(i, time_point, 'HERTZ')
                if not np.isnan(f_value) and f_value > 0:
                    formants_dict[f'f{i}'] = float(f_value)
                else:
                    formants_dict[f'f{i}'] = 0.0
            
            return formants_dict
        except Exception as e:
            print(f"Error extracting formants: {e}")
            return {f'f{i}': 0.0 for i in range(1, self.config.formant_number + 1)}
    
    def extract_mfccs(self, y: np.ndarray, sr: int) -> list:
        """Extract MFCC features"""
        try:
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.config.mfcc_number,
                n_fft=2048,
                hop_length=512
            )
            # Return mean MFCCs across time
            return np.mean(mfccs, axis=1).tolist()
        except Exception as e:
            print(f"Error extracting MFCCs: {e}")
            return [0.0] * self.config.mfcc_number
    
    def extract_voice_quality(self, sound: parselmouth.Sound) -> VoiceQuality:
        try:
            point_process = parselmouth.praat.call(
                sound, "To PointProcess (periodic, cc)",
                self.config.pitch_min_f0,
                self.config.pitch_max_f0
            )

            jitter = parselmouth.praat.call(
                point_process, "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3
            )

            shimmer = parselmouth.praat.call(
                [sound, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            harmonicity = sound.to_harmonicity_cc()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

            return VoiceQuality(
                jitter_percent=float(jitter * 100) if jitter is not None and not np.isnan(jitter) else None,
                shimmer_db=float(shimmer) if shimmer is not None and not np.isnan(shimmer) else None,
                harmonic_to_noise_ratio=float(hnr) if hnr is not None and not np.isnan(hnr) else None
                )
                
            

        except Exception as e:
            print(f"Error extracting voice quality: {e}")
            return VoiceQuality(jitter_percent=None, shimmer_db=None, harmonic_to_noise_ratio=None)

    def extract_waveform_features(
        self,
        y: np.ndarray,
        sr: int,
        session_id: str,
        *,
        sound: Optional[parselmouth.Sound] = None,
        segment_id: Optional[str] = None,
        role: Optional[str] = None,
        speaker_id: Optional[str] = None,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        source_filename: Optional[str] = None,
    ) -> FeatureExtractionResult:
        """Extract all features from an in-memory waveform."""
        import datetime

        if sound is None:
            sound = parselmouth.Sound(y, sr)

        duration = len(y) / sr if sr else 0.0

        # Quality metrics
        snr = self.estimate_snr(y, sr)
        noise_level = self.assess_noise_level(snr)

        recording_quality = RecordingQuality(
            duration_seconds=duration,
            snr_db=snr,
            background_noise_level=noise_level,
            sample_rate=sr
        )

        # Feature extraction
        pitch_features = self.extract_pitch_features(sound)
        formants = self.extract_formants(sound)
        mfccs = self.extract_mfccs(y, sr)
        voice_quality = self.extract_voice_quality(sound)

        # Compile features
        features = {
            "pitch": asdict(pitch_features),
            "timbre": {
                "mfccs": mfccs,
                "formants": formants
            },
            "voice_quality": asdict(voice_quality)
        }

        return FeatureExtractionResult(
            session_id=session_id,
            recording_quality=asdict(recording_quality),
            features=features,
            processing_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            segment_id=segment_id,
            role=role,
            speaker_id=speaker_id,
            start_sec=start_sec,
            end_sec=end_sec,
            source_filename=source_filename,
        )

    def extract_features(self, audio_data: bytes, session_id: str, original_filename: str = None) -> FeatureExtractionResult:
        """Main method to extract all features from raw audio bytes."""
        y, sr, sound = self.load_audio(audio_data, original_filename=original_filename)
        return self.extract_waveform_features(
            y,
            sr,
            session_id,
            sound=sound,
            source_filename=original_filename,
        )

    def get_genetic_features(self, y: np.ndarray, sr: int, sound: parselmouth.Sound = None) -> dict:
        """Fetch features for genetic model"""
        if sound is None:
            sound = parselmouth.Sound(y, sr)

        # Pitch features
        pitch = self.extract_pitch_features(sound)

        # Voice quality
        voice_quality = self.extract_voice_quality(sound)

        return {
            'pitch_mean': pitch.mean_f0_hz,
            'pitch_variability': pitch.variability,
            'jitter': voice_quality.jitter_percent or 0,
            'shimmer': voice_quality.shimmer_db or 0,
            'hnr': voice_quality.harmonic_to_noise_ratio or 0,
        }