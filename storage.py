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
    return datetime.datetime.utcnow().isoformat() + "Z"


def _guess_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".bin"
    suffix = Path(filename).suffix.lower()
    return suffix if suffix else ".bin"


@dataclass
class StoredAudioMetadata:
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


class LocalRawAudioStorage:
    """Filesystem-backed storage for raw audio outside git."""

    def __init__(self, root_dir: Optional[Path | str] = None):
        self.root_dir = Path(
            root_dir
            or os.getenv("VOICE_GENETICS_STORAGE_DIR", "storage")
        )
        self.audio_root = self.root_dir / "raw_audio"
        self.audio_root.mkdir(parents=True, exist_ok=True)

    def _record_dir(self, audio_id: str) -> Path:
        return self.audio_root / audio_id

    def _audio_path(self, audio_id: str, original_filename: Optional[str]) -> Path:
        return self._record_dir(audio_id) / f"source{_guess_suffix(original_filename)}"

    def _metadata_path(self, audio_id: str) -> Path:
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
        audio_id = uuid.uuid4().hex
        record_dir = self._record_dir(audio_id)
        record_dir.mkdir(parents=True, exist_ok=True)

        audio_path = self._audio_path(audio_id, original_filename)
        audio_path.write_bytes(audio_bytes)

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
        audio_bytes = upload_file.file.read()
        return self.save_bytes(
            audio_bytes,
            original_filename=upload_file.filename,
            content_type=getattr(upload_file, "content_type", None),
            session_id=session_id,
            purpose=purpose,
        )

    def load_metadata(self, audio_id: str) -> StoredAudioMetadata:
        metadata_path = self._metadata_path(audio_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Audio metadata not found for audio_id={audio_id}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return StoredAudioMetadata(**payload)

    def load_bytes(self, audio_id: str) -> bytes:
        metadata = self.load_metadata(audio_id)
        audio_path = Path(metadata.storage_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at {metadata.storage_path}")
        return audio_path.read_bytes()

    def delete(self, audio_id: str) -> None:
        record_dir = self._record_dir(audio_id)
        if not record_dir.exists():
            return
        for child in record_dir.iterdir():
            if child.is_file():
                child.unlink()
        record_dir.rmdir()
