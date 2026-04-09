from pydantic import BaseModel
from typing import Optional

class ExtractionConfig(BaseModel):
    """Configurable feature extraction parameters"""
    
    # Pitch extraction parameters
    pitch_min_f0: Optional[float] = 75.0
    pitch_max_f0: Optional[float] = 300.0
    pitch_unit: Optional[str] = "Hz"
    
    # Formant extraction parameters
    formant_max_frequency: Optional[float] = 5500.0
    formant_number: Optional[int] = 4
    
    # MFCC parameters
    mfcc_number: Optional[int] = 13
    mfcc_use_energy: Optional[bool] = True
    
    # Voice quality parameters
    jitter_method: Optional[str] = "RAP"  # Relative Average Perturbation
    shimmer_method: Optional[str] = "APQ3"  # Amplitude Perturbation Quotient
    
    # Quality metrics
    min_snr_db: Optional[float] = 20.0
    min_duration_seconds: Optional[float] = 0.5
    
    # Processing parameters
    target_sample_rate: Optional[int] = 16000

    # Dialogue / diarization parameters
    vad_top_db: Optional[float] = 28.0
    min_turn_duration_seconds: Optional[float] = 0.35
    merge_gap_seconds: Optional[float] = 0.25
    max_dialogue_speakers: Optional[int] = 8
    min_speech_energy_threshold: Optional[float] = 0.0
    barge_in_gap_seconds: Optional[float] = 0.35
    barge_in_trim_seconds: Optional[float] = 0.25
    music_min_duration_seconds: Optional[float] = 6.0
    music_min_gap_seconds: Optional[float] = 0.75
    music_min_score: Optional[int] = 4
    speaker_role_similarity_threshold: Optional[float] = 0.78
    music_role_similarity_threshold: Optional[float] = 0.62
    dialogue_cluster_distance_threshold: Optional[float] = 0.04

    class Config:
        arbitrary_types_allowed = True

# Default configuration
DEFAULT_CONFIG = ExtractionConfig()
