from __future__ import annotations

import datetime
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def _utc_now() -> str:
    """
    Returns the current UTC time in ISO format with 'Z' suffix.
    """
    return datetime.datetime.utcnow().isoformat() + "Z"


def _guess_suffix(filename: Optional[str]) -> str:
    """
    Extract file extension from filename or return default '.bin'.
    """
    if not filename:
        return ".bin"
    suffix = Path(filename).suffix.lower()
    return suffix if suffix else ".bin"


@dataclass
class StoredAudioMetadata:
    """
    Metadata for a stored audio file.
    """
    audio_id: str  # Unique identifier (UUID hex)
    original_filename: str  # Original name as uploaded
    content_type: Optional[str]  # MIME type (e.g., 'audio/wav')
    byte_size: int  # Size in bytes
    sha256: str  # SHA256 hash for integrity checks
    created_at: str  # UTC timestamp of upload
    storage_backend: str  # 'local_filesystem' (extensible for S3 etc.)
    storage_path: str  # Absolute path to stored audio file
    session_id: Optional[str] = None  # Optional session grouping
    purpose: Optional[str] = None  # Optional tag (e.g., 'dialogue', 'analysis')


class LocalRawAudioStorage:
    """
    Filesystem-backed storage for raw audio files.
    """

    def __init__(self, root_dir: Optional[Path | str] = None):
        """
        Initialize storage with optional custom root directory.
        """
        self.root_dir = Path(
            root_dir
            or os.getenv("VOICE_GENETICS_STORAGE_DIR", "storage")
        )
        self.audio_root = self.root_dir / "raw_audio"
        self.audio_root.mkdir(parents=True, exist_ok=True)

    def _record_dir(self, audio_id: str) -> Path:
        """Returns the directory path for a specific audio_id."""
        return self.audio_root / audio_id

    def _audio_path(self, audio_id: str, original_filename: Optional[str]) -> Path:
        """
        Returns the full path for the audio file.
        """
        return self._record_dir(audio_id) / f"source{_guess_suffix(original_filename)}"

    def _metadata_path(self, audio_id: str) -> Path:
        """Returns the full path for the metadata JSON file."""
        return self._record_dir(audio_id) / "metadata.json"

    def save_bytes(
            self,
            audio_bytes: bytes,
            *,
            original_filename: Optional[str] = None,
            content_type: Optional[str] = None,
            session_id: Optional[str] = None,
            purpose: Optional[str] = None,
    ) -> StoredAudioMetadata:
        """
        Save raw audio bytes to storage and return metadata.

        This is the primary storage method. It:
        1. Generates a unique audio_id (UUID)
        2. Creates a directory for that ID
        3. Writes the audio bytes to disk
        4. Computes SHA256 hash for integrity
        5. Saves metadata as JSON
        6. Returns metadata object
        """
        audio_id = uuid.uuid4().hex
        record_dir = self._record_dir(audio_id)
        record_dir.mkdir(parents=True, exist_ok=True)

        # Save audio file
        audio_path = self._audio_path(audio_id, original_filename)
        audio_path.write_bytes(audio_bytes)

        # Create metadata record
        metadata = StoredAudioMetadata(
            audio_id=audio_id,
            original_filename=original_filename or audio_path.name,
            content_type=content_type,
            byte_size=len(audio_bytes),
            sha256=hashlib.sha256(audio_bytes).hexdigest(),
            created_at=_utc_now(),
            storage_backend="local_filesystem",
            storage_path=str(audio_path),
            session_id=session_id,
            purpose=purpose,
        )

        # Save metadata as JSON
        self._metadata_path(audio_id).write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata

    def save_upload(
            self,
            upload_file,
            *,
            session_id: Optional[str] = None,
            purpose: Optional[str] = None,
    ) -> StoredAudioMetadata:
        """
        Convenience method to save a Streamlit UploadedFile object.
        """
        audio_bytes = upload_file.file.read()
        return self.save_bytes(
            audio_bytes,
            original_filename=upload_file.filename,
            content_type=getattr(upload_file, "content_type", None),
            session_id=session_id,
            purpose=purpose,
        )

    def load_metadata(self, audio_id: str) -> StoredAudioMetadata:
        """
        Load metadata for a given audio_id.
        """
        metadata_path = self._metadata_path(audio_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Audio metadata not found for audio_id={audio_id}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return StoredAudioMetadata(**payload)

    def load_bytes(self, audio_id: str) -> bytes:
        """
        Load raw audio bytes for a given audio_id.
        """
        metadata = self.load_metadata(audio_id)
        audio_path = Path(metadata.storage_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {metadata.storage_path}")
        return audio_path.read_bytes()

    def delete(self, audio_id: str) -> None:
        """
        Permanently delete audio file and metadata for a given audio_id.
        """
        record_dir = self._record_dir(audio_id)
        if not record_dir.exists():
            return
        # Delete all files in the directory
        for child in record_dir.iterdir():
            if child.is_file():
                child.unlink()
        # Remove the empty directory
        record_dir.rmdir()