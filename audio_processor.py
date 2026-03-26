import parselmouth
import numpy as np
import librosa
import io
import tempfile
import os
from typing import Dict, Any, Optional
import scipy.signal as signal
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
    jitter_percent: float
    shimmer_db: float
    harmonic_to_noise_ratio: Optional[float] = None

@dataclass
class FeatureExtractionResult:
    session_id: str
    recording_quality: Dict[str, Any]
    features: Dict[str, Any]
    processing_timestamp: str

class VoiceFeatureExtractor:
    def __init__(self, config=None):
        self.config = config
        if self.config is None:
            from config import DEFAULT_CONFIG
            self.config = DEFAULT_CONFIG
    
    def load_audio(self, audio_data: bytes, original_filename: str = None) -> tuple:
        """Load audio from bytes and resample if needed"""
        try:
            # Save to temporary file for Parselmouth
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            # Load with librosa for initial processing
            y, sr = librosa.load(tmp_path, sr=self.config.target_sample_rate)
            
            # Also load with Parselmouth for advanced analysis
            sound = parselmouth.Sound(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return y, sr, sound
            
        except Exception as e:
            raise Exception(f"Error loading audio: {str(e)}")
    
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
    def extract_features(self, audio_data: bytes, session_id: str) -> FeatureExtractionResult:
        """Main method to extract all features"""
        import datetime
        
        # Load audio
        y, sr, sound = self.load_audio(audio_data)
        duration = len(y) / sr
        
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
            processing_timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )