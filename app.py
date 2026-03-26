from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging
import uuid

from audio_processor import VoiceFeatureExtractor
from config import ExtractionConfig, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Feature Extractor",
    description="Extract acoustic features from voice recordings using Parselmouth",
    version="1.0.0"
)

# Initialize extractor with default config
extractor = VoiceFeatureExtractor(DEFAULT_CONFIG)

class FeatureResponse(BaseModel):
    session_id: str
    recording_quality: Dict[str, Any]
    features: Dict[str, Any]
    processing_timestamp: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "voice-genetics"}

@app.post("/extract", response_model=FeatureResponse)
async def extract_features(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a)"),
    session_id: Optional[str] = Query(None, description="Optional session ID"),
    min_duration: Optional[float] = Query(None, description="Minimum duration in seconds"),
    max_formants: Optional[int] = Query(None, description="Number of formants to extract"),
    mfcc_count: Optional[int] = Query(None, description="Number of MFCC coefficients")
):
    """
    Extract acoustic features from uploaded audio file.
    
    Returns JSON with extracted features, quality metrics, and no raw audio data.
    """
    
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.m4a'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Read audio data
        audio_data = await file.read()
        
        # Validate file size (max 50MB)
        if len(audio_data) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")
        
        # Create temporary config with overrides if needed
        config = DEFAULT_CONFIG
        if min_duration is not None:
            config.min_duration_seconds = min_duration
        if max_formants is not None:
            config.formant_number = max_formants
        if mfcc_count is not None:
            config.mfcc_number = mfcc_count
        
        # Update extractor config if changed
        if config != DEFAULT_CONFIG:
            extractor.config = config
        
        # Extract features
        result = extractor.extract_features(audio_data, session_id)
        
        # Validate minimum duration
        if result.recording_quality["duration_seconds"] < config.min_duration_seconds:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Audio too short. Minimum duration: {config.min_duration_seconds}s",
                    "actual_duration": result.recording_quality["duration_seconds"]
                }
            )
        
        # Convert to response model
        response = FeatureResponse(
            session_id=result.session_id,
            recording_quality=result.recording_quality,
            features=result.features,
            processing_timestamp=result.processing_timestamp
        )
        
        logger.info(f"Successfully extracted features for session: {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@app.post("/extract/batch")
async def extract_batch_features(
    files: list[UploadFile] = File(..., description="Multiple audio files"),
    session_ids: Optional[str] = Query(None, description="Comma-separated session IDs")
):
    """
    Extract features from multiple audio files.
    """
    results = []
    session_list = session_ids.split(',') if session_ids else []
    
    for idx, file in enumerate(files):
        try:
            session_id = session_list[idx] if idx < len(session_list) else str(uuid.uuid4())
            audio_data = await file.read()
            result = extractor.extract_features(audio_data, session_id)
            results.append(result.__dict__)
        except Exception as e:
            results.append({
                "error": f"Failed to process {file.filename}: {str(e)}",
                "filename": file.filename
            })
    
    return {"results": results, "total_processed": len(results)}

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return extractor.config.dict()

@app.post("/config")
async def update_config(config: ExtractionConfig):
    """Update feature extraction configuration"""
    extractor.config = config
    return {"status": "updated", "config": config.dict()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)