import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import logging
import uuid

from audio_processor import VoiceFeatureExtractor
from config import ExtractionConfig, DEFAULT_CONFIG
from dialogue_processor import DialogueProcessor, DialogueSegmentSpec
from storage import LocalRawAudioStorage, StoredAudioMetadata
from genetic_model import predictor

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Voice Feature Extractor",
    description="Extract acoustic features from voice recordings using Parselmouth",
    version="1.0.0"
)

# Shared instances to avoid reloading on each request
shared_extractor = VoiceFeatureExtractor(DEFAULT_CONFIG.model_copy(deep=True))
raw_audio_storage = LocalRawAudioStorage()


def build_runtime_config(
        *,
        min_duration: Optional[float] = None,
        max_formants: Optional[int] = None,
        mfcc_count: Optional[int] = None,
        pitch_min_f0: Optional[float] = None,
        pitch_max_f0: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
        vad_top_db: Optional[float] = None,
        min_turn_duration_seconds: Optional[float] = None,
        merge_gap_seconds: Optional[float] = None,
) -> ExtractionConfig:
    """
    Build an ExtractionConfig with optional overrides from query parameters.

    This allows clients to customize feature extraction parameters without
    modifying the server configuration.
    """
    config = DEFAULT_CONFIG.model_copy(deep=True)
    if min_duration is not None:
        config.min_duration_seconds = min_duration
    if max_formants is not None:
        config.formant_number = max_formants
    if mfcc_count is not None:
        config.mfcc_number = mfcc_count
    if pitch_min_f0 is not None:
        config.pitch_min_f0 = pitch_min_f0
    if pitch_max_f0 is not None:
        config.pitch_max_f0 = pitch_max_f0
    if target_sample_rate is not None:
        config.target_sample_rate = target_sample_rate
    if vad_top_db is not None:
        config.vad_top_db = vad_top_db
    if min_turn_duration_seconds is not None:
        config.min_turn_duration_seconds = min_turn_duration_seconds
    if merge_gap_seconds is not None:
        config.merge_gap_seconds = merge_gap_seconds
    return config


def parse_dialogue_segments(segments_json: str) -> List[DialogueSegmentSpec]:
    """
    Parse JSON array into list of DialogueSegmentSpec objects.

    This validates the manifest format and converts string values
    to appropriate types (float, bool, etc.).
    """
    payload = json.loads(segments_json)
    if not isinstance(payload, list):
        raise ValueError("segments_json must be a JSON array")

    segments: List[DialogueSegmentSpec] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Segment at index {index} must be an object")
        is_barge_in_value = item.get("is_barge_in")
        if isinstance(is_barge_in_value, str):
            is_barge_in = is_barge_in_value.strip().lower() in {"1", "true", "yes", "y"}
        elif is_barge_in_value is None:
            is_barge_in = None
        else:
            is_barge_in = bool(is_barge_in_value)
        segments.append(
            DialogueSegmentSpec(
                segment_id=str(item.get("segment_id") or f"segment_{index:03d}"),
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                analysis_start_sec=float(item["analysis_start_sec"]) if item.get("analysis_start_sec") not in (None,
                                                                                                               "") else None,
                analysis_end_sec=float(item["analysis_end_sec"]) if item.get("analysis_end_sec") not in (None,
                                                                                                         "") else None,
                role=item.get("role"),
                speaker_id=item.get("speaker_id"),
                text=item.get("text"),
                turn_type=item.get("turn_type"),
                is_barge_in=is_barge_in,
                source=item.get("source", "manifest"),
            )
        )
    return segments


# Pydantic models for request/response validation
class FeatureResponse(BaseModel):
    """Response model for feature extraction endpoint."""
    session_id: str
    recording_quality: Dict[str, Any]
    features: Dict[str, Any]
    processing_timestamp: str


class DialogueSegmentRequest(BaseModel):
    """Request model for a single dialogue segment."""
    segment_id: str
    start_sec: float
    end_sec: float
    analysis_start_sec: Optional[float] = None
    analysis_end_sec: Optional[float] = None
    role: Optional[str] = None
    speaker_id: Optional[str] = None
    text: Optional[str] = None
    turn_type: Optional[str] = None
    is_barge_in: Optional[bool] = None
    source: Optional[str] = "manifest"


class DialogueAnalysisResponse(BaseModel):
    """Response model for dialogue analysis."""
    session_id: str
    source_filename: Optional[str]
    recording_quality: Dict[str, Any]
    diarization: Dict[str, Any]
    speaker_summary: Dict[str, Any]
    user_summary: Dict[str, Any]
    turns: List[Dict[str, Any]]
    processing_timestamp: str


class AudioUploadResponse(BaseModel):
    """Response model for audio upload endpoint."""
    audio_id: str
    original_filename: str
    content_type: Optional[str]
    byte_size: int
    sha256: str
    created_at: str
    storage_backend: str
    storage_path: str
    session_id: Optional[str] = None
    purpose: Optional[str] = None


def _metadata_response(metadata: StoredAudioMetadata) -> AudioUploadResponse:
    """Convert internal metadata to API response model."""
    return AudioUploadResponse(**metadata.__dict__)


def _turns_to_manifest_csv(turns: List[Dict[str, Any]], source_filename: Optional[str]) -> str:
    """Convert diarization turns to CSV string for download."""
    extractor = DialogueProcessor(shared_extractor)
    return extractor.turns_to_manifest_csv(turns, source_filename=source_filename)

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns simple status to confirm service is running.
    """
    return {"status": "healthy", "service": "voice-genetics"}


@app.post("/audio/upload", response_model=AudioUploadResponse)
async def upload_audio(
        file: UploadFile = File(..., description="Audio file to store in backend storage"),
        session_id: Optional[str] = Query(None, description="Optional session ID"),
        purpose: Optional[str] = Query(None, description="Purpose tag such as dialogue or extraction"),
):
    """
    Upload and store raw audio file in backend storage.

    Returns metadata including audio_id for later retrieval.
    This is useful for keeping raw audio outside git and
    for analyzing stored files without re-uploading.
    """
    # Validate file format
    if not any(file.filename.lower().endswith(ext) for ext in {".wav", ".mp3", ".m4a"}):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: wav, mp3, m4a")

    audio_data = await file.read()
    if len(audio_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

    metadata = raw_audio_storage.save_bytes(
        audio_data,
        original_filename=file.filename,
        content_type=file.content_type,
        session_id=session_id,
        purpose=purpose,
    )
    return _metadata_response(metadata)


@app.get("/audio/{audio_id}/metadata", response_model=AudioUploadResponse)
async def get_audio_metadata(audio_id: str):
    """
    Retrieve metadata for a stored audio file without downloading the file itself.
    Useful for checking file info before analysis.
    """
    try:
        metadata = raw_audio_storage.load_metadata(audio_id)
        return _metadata_response(metadata)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/dialogue/manifest", response_class=PlainTextResponse)
async def build_dialogue_manifest(
        file: UploadFile = File(..., description="Conversation audio file (wav, mp3, m4a)"),
        session_id: Optional[str] = Query(None, description="Optional session ID"),
        vad_top_db: Optional[float] = Query(None, description="Energy threshold for speech turn detection"),
        min_turn_duration_seconds: Optional[float] = Query(None, description="Minimum turn duration in seconds"),
        merge_gap_seconds: Optional[float] = Query(None, description="Merge silence gaps smaller than this"),
):
    """
    Generate a dialogue_manifest.csv from conversation audio.

    This runs diarization and returns a CSV that users can edit
    to correct role assignments before role-aware analysis.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    if not any(file.filename.lower().endswith(ext) for ext in {".wav", ".mp3", ".m4a"}):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: wav, mp3, m4a")

    audio_data = await file.read()
    if len(audio_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

    try:
        extractor = VoiceFeatureExtractor(
            build_runtime_config(
                vad_top_db=vad_top_db,
                min_turn_duration_seconds=min_turn_duration_seconds,
                merge_gap_seconds=merge_gap_seconds,
            )
        )
        processor = DialogueProcessor(extractor)
        diarization = processor.diarize_dialogue(audio_data, session_id, original_filename=file.filename)
        csv_text = _turns_to_manifest_csv(diarization.turns, file.filename)
        headers = {
            "Content-Disposition": 'attachment; filename="dialogue_manifest.csv"',
        }
        return PlainTextResponse(content=csv_text, media_type="text/csv", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building dialogue manifest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dialogue manifest generation failed: {str(e)}")

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

    This is the main endpoint for single recording analysis.
    Returns JSON with extracted features, quality metrics, and no raw audio data.

    Features extracted:
    - Pitch (mean, min, max, variability)
    - Voice quality (jitter, shimmer, HNR)
    - Timbre (formants, MFCCs)
    - Recording quality (duration, SNR, noise level)
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
        extractor = VoiceFeatureExtractor(
            build_runtime_config(
                min_duration=min_duration,
                max_formants=max_formants,
                mfcc_count=mfcc_count,
            )
        )

        # Read audio data
        audio_data = await file.read()

        # Validate file size (max 50MB)
        if len(audio_data) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

        # Extract features
        result = extractor.extract_features(audio_data, session_id, original_filename=file.filename)

        # Validate minimum duration
        if result.recording_quality["duration_seconds"] < extractor.config.min_duration_seconds:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Audio too short. Minimum duration: {extractor.config.min_duration_seconds}s",
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@app.post("/dialogue/diarize", response_model=DialogueAnalysisResponse)
async def diarize_dialogue(
        file: UploadFile = File(..., description="Conversation audio file (wav, mp3, m4a)"),
        session_id: Optional[str] = Query(None, description="Optional session ID"),
        vad_top_db: Optional[float] = Query(None, description="Energy threshold for speech turn detection"),
        min_turn_duration_seconds: Optional[float] = Query(None, description="Minimum turn duration in seconds"),
        merge_gap_seconds: Optional[float] = Query(None, description="Merge silence gaps smaller than this"),
):
    """
    Run diarization on conversation audio.

    Returns speech turns with speaker clusters and semantic role labels
    (assistant_1, user, assistant_2, music).

    This is the first step in the dialogue analysis pipeline.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    if not any(file.filename.lower().endswith(ext) for ext in {".wav", ".mp3", ".m4a"}):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: wav, mp3, m4a")

    audio_data = await file.read()
    if len(audio_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

    try:
        extractor = VoiceFeatureExtractor(
            build_runtime_config(
                vad_top_db=vad_top_db,
                min_turn_duration_seconds=min_turn_duration_seconds,
                merge_gap_seconds=merge_gap_seconds,
            )
        )
        processor = DialogueProcessor(extractor)
        result = processor.diarize_dialogue(audio_data, session_id, original_filename=file.filename)
        return DialogueAnalysisResponse(**result.__dict__)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error diarizing dialogue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dialogue diarization failed: {str(e)}")


# ==================== Role-Aware Dialogue Analysis ====================

@app.post("/dialogue/analyze", response_model=DialogueAnalysisResponse)
async def analyze_dialogue(
        file: UploadFile = File(..., description="Conversation audio file (wav, mp3, m4a)"),
        segments_json: str = Form(..., description="JSON array of dialogue segments with roles"),
        session_id: Optional[str] = Query(None, description="Optional session ID"),
        min_duration: Optional[float] = Query(None, description="Minimum duration in seconds"),
        max_formants: Optional[int] = Query(None, description="Number of formants to extract"),
        mfcc_count: Optional[int] = Query(None, description="Number of MFCC coefficients"),
):
    """
    Analyze labeled dialogue and extract user-only features.

    This endpoint expects a manifest with role labels.
    It extracts features only for segments marked as 'user'
    and returns aggregated summary statistics.

    This is the second step in the dialogue analysis pipeline.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    if not any(file.filename.lower().endswith(ext) for ext in {".wav", ".mp3", ".m4a"}):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: wav, mp3, m4a")

    audio_data = await file.read()
    if len(audio_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

    try:
        segments = parse_dialogue_segments(segments_json)
        extractor = VoiceFeatureExtractor(
            build_runtime_config(
                min_duration=min_duration,
                max_formants=max_formants,
                mfcc_count=mfcc_count,
            )
        )
        processor = DialogueProcessor(extractor)
        result = processor.analyze_labeled_dialogue(
            audio_data,
            session_id,
            segments,
            original_filename=file.filename,
            strict_roles=True,
        )
        return DialogueAnalysisResponse(**result.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing dialogue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dialogue analysis failed: {str(e)}")


@app.post("/dialogue/analyze-stored", response_model=DialogueAnalysisResponse)
async def analyze_stored_dialogue(
        audio_id: str = Form(..., description="Stored audio identifier"),
        segments_json: str = Form(..., description="JSON array of dialogue segments with roles"),
        session_id: Optional[str] = Query(None, description="Optional session ID"),
        min_duration: Optional[float] = Query(None, description="Minimum duration in seconds"),
        max_formants: Optional[int] = Query(None, description="Number of formants to extract"),
        mfcc_count: Optional[int] = Query(None, description="Number of MFCC coefficients"),
):
    """
    Analyze previously stored dialogue audio.

    Same as /dialogue/analyze but uses audio_id instead of direct upload.
    Useful for re-analyzing stored files without re-uploading.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        metadata = raw_audio_storage.load_metadata(audio_id)
        audio_data = raw_audio_storage.load_bytes(audio_id)
        segments = parse_dialogue_segments(segments_json)
        extractor = VoiceFeatureExtractor(
            build_runtime_config(
                min_duration=min_duration,
                max_formants=max_formants,
                mfcc_count=mfcc_count,
            )
        )
        processor = DialogueProcessor(extractor)
        result = processor.analyze_labeled_dialogue(
            audio_data,
            session_id,
            segments,
            original_filename=metadata.original_filename,
            strict_roles=True,
        )
        return DialogueAnalysisResponse(**result.__dict__)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing stored dialogue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stored dialogue analysis failed: {str(e)}")


@app.post("/extract/batch")
async def extract_batch_features(
        files: list[UploadFile] = File(..., description="Multiple audio files"),
        session_ids: Optional[str] = Query(None, description="Comma-separated session IDs")
):
    """
    Extract features from multiple audio files in batch mode.

    Useful for processing large datasets or running experiments.
    Returns results for all files, with errors reported per file.
    """
    results = []
    session_list = session_ids.split(',') if session_ids else []

    for idx, file in enumerate(files):
        try:
            session_id = session_list[idx] if idx < len(session_list) else str(uuid.uuid4())
            audio_data = await file.read()
            extractor = VoiceFeatureExtractor(DEFAULT_CONFIG.model_copy(deep=True))
            result = extractor.extract_features(audio_data, session_id, original_filename=file.filename)
            results.append(result.__dict__)
        except Exception as e:
            results.append({
                "error": f"Failed to process {file.filename}: {str(e)}",
                "filename": file.filename
            })

    return {"results": results, "total_processed": len(results)}


@app.post("/genetic/predict")
async def predict_genotype(
        file: UploadFile = File(..., description="Audio file for genetic prediction"),
):
    """
    Predict genotype for rs11046212 (gene ABCC9) from voice features.

    Based on GWAS study by Gisladottir et al. (Science Advances 2023).
    Returns predicted genotype (CC, CT, TT) with confidence probabilities.

    This endpoint integrates the genetic model with the feature extractor.
    """
    # Checking format
    if not any(file.filename.lower().endswith(ext) for ext in {".wav", ".mp3", ".m4a"}):
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Read audio
    audio_data = await file.read()
    if len(audio_data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        # Fetch features using the shared extractor
        y, sr, sound = shared_extractor.load_audio(audio_data, original_filename=file.filename)

        # Get genetically relevant features (pitch_mean, variability, jitter, shimmer, hnr)
        genetic_features = shared_extractor.get_genetic_features(y, sr, sound)

        # Predict genotype using the trained Random Forest model
        prediction = predictor.predict(genetic_features)

        # Add extracted features to response for transparency
        prediction["extracted_features"] = genetic_features

        return prediction

    except Exception as e:
        logger.error(f"Error in genetic prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current feature extraction configuration."""
    return shared_extractor.config.dict()

@app.post("/config")
async def update_config(config: ExtractionConfig):
    """Update feature extraction configuration dynamically."""
    shared_extractor.config = config
    return {"status": "updated", "config": config.dict()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)