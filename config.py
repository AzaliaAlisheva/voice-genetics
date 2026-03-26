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
    
    class Config:
        arbitrary_types_allowed = True

# Default configuration
DEFAULT_CONFIG = ExtractionConfig()