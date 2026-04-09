#!/usr/bin/env python3
"""Batch-analyze conversation recordings using a segment-level dialogue manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Ensure project root imports work when this script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_processor import VoiceFeatureExtractor
from config import DEFAULT_CONFIG
from dialogue_processor import DialogueProcessor, DialogueSegmentSpec


LOGGER = logging.getLogger("voice_dialogue_batch")
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze dialogue recordings using a segment-level manifest."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data"), help="Directory with audio files.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest with one row per segment.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for results.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of recordings.")
    return parser


def load_manifest(manifest_path: Path) -> Dict[str, List[DialogueSegmentSpec]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    grouped: Dict[str, List[DialogueSegmentSpec]] = defaultdict(list)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "segment_id", "start_sec", "end_sec", "role"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest must contain columns: filename, segment_id, start_sec, end_sec, role"
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            grouped[filename].append(
                DialogueSegmentSpec(
                    segment_id=str(row.get("segment_id") or uuid.uuid4().hex[:8]),
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    analysis_start_sec=float(row["analysis_start_sec"]) if row.get("analysis_start_sec") not in (None, "") else None,
                    analysis_end_sec=float(row["analysis_end_sec"]) if row.get("analysis_end_sec") not in (None, "") else None,
                    role=row.get("role"),
                    speaker_id=row.get("speaker_id"),
                    text=row.get("text"),
                    turn_type=row.get("turn_type"),
                    is_barge_in=str(row.get("is_barge_in", "")).strip().lower() in {"1", "true", "yes", "y"},
                    source=row.get("source", "manifest"),
                )
            )

    return grouped


def flatten_feature_map(feature_map: Dict[str, float], meta: Dict[str, str]) -> Dict[str, object]:
    row: Dict[str, object] = dict(meta)
    row.update(feature_map)
    return row


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return 1

    grouped_manifest = load_manifest(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = VoiceFeatureExtractor(DEFAULT_CONFIG.model_copy(deep=True))
    processor = DialogueProcessor(extractor)

    run_id = uuid.uuid4().hex[:8]
    summary_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    filenames = sorted(grouped_manifest.keys())
    if args.limit is not None:
        filenames = filenames[: args.limit]

    for filename in filenames:
        audio_path = args.input_dir / filename
        if not audio_path.exists():
            failures.append({"filename": filename, "error": "audio file not found"})
            continue
        if audio_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            failures.append({"filename": filename, "error": "unsupported file format"})
            continue

        try:
            audio_data = audio_path.read_bytes()
            result = processor.analyze_labeled_dialogue(
                audio_data,
                session_id=str(uuid.uuid4()),
                segments=grouped_manifest[filename],
                original_filename=filename,
                strict_roles=True,
            )

            feature_map = result.user_summary["feature_map"]
            summary_rows.append(
                flatten_feature_map(
                    feature_map,
                    {
                        "filename": filename,
                        "turn_count": len(result.turns),
                        "user_turn_count": result.diarization["user_turn_count"],
                        "processing_timestamp": result.processing_timestamp,
                    },
                )
            )

            json_path = output_dir / f"{audio_path.stem}_{run_id}.json"
            json_path.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info("Processed %s", filename)
        except Exception as exc:  # noqa: BLE001
            failures.append({"filename": filename, "error": str(exc)})
            LOGGER.exception("Failed to process %s", filename)

    if not summary_rows:
        LOGGER.error("No dialogue recordings were processed successfully.")
        return 1

    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    csv_path = output_dir / f"dialogue_features_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    failures_path = output_dir / f"dialogue_failures_{run_id}.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved CSV: %s", csv_path)
    LOGGER.info("Saved failures: %s", failures_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
