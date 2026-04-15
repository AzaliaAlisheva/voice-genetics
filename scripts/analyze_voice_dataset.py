#!/usr/bin/env python3
"""Batch-analyze voice recordings from a local dataset directory.

This script is intended for local research workflows:
- keep raw audio outside git
- optionally use a manifest with speaker labels
- extract acoustic features only for the requested speaker type
- export a flat table for downstream analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root imports work when this script is executed directly.
# This allows the script to be run from anywhere while correctly resolving imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("voice_dataset_analyzer")
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a"}


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser for the batch dataset analysis tool.
    """
    parser = argparse.ArgumentParser(
        description="Extract voice features from a directory of recordings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing audio files to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where results (CSV and JSON) will be saved.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest with columns: filename,speaker_type. "
             "Use this to keep only user audio and skip assistant audio.",
    )
    parser.add_argument(
        "--speaker-type",
        choices=["user", "assistant", "all"],
        default="user",
        help="Which speaker type to analyze. Requires a manifest unless set to 'all'.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum accepted duration in seconds. Shorter files are skipped.",
    )
    parser.add_argument(
        "--pitch-min-f0",
        type=float,
        default=75.0,
        help="Minimum pitch floor in Hz (for voice pitch detection).",
    )
    parser.add_argument(
        "--pitch-max-f0",
        type=float,
        default=300.0,
        help="Maximum pitch ceiling in Hz (for voice pitch detection).",
    )
    parser.add_argument(
        "--formant-number",
        type=int,
        default=4,
        help="Number of formants to extract (F1, F2, F3, F4).",
    )
    parser.add_argument(
        "--mfcc-number",
        type=int,
        default=13,
        help="Number of MFCC coefficients to extract.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for processing (resampling if needed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files to process (useful for testing).",
    )
    return parser


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load and parse the CSV manifest file.

    The manifest maps filenames to speaker_type labels and optional metadata.
    This allows filtering by speaker type (user vs assistant).
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    mapping: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "speaker_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest must contain columns: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            speaker_type = (row.get("speaker_type") or "").strip().lower()
            if filename:
                mapping[filename] = {
                    "speaker_type": speaker_type,
                    "session_id": (row.get("session_id") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }

    return mapping


def make_config(args: argparse.Namespace, base_config: Any) -> Any:
    """
    Create a custom configuration from command-line arguments.

    This allows overriding default extraction parameters without
    modifying the global configuration file.
    """
    config = base_config.model_copy(deep=True)
    config.min_duration_seconds = args.min_duration
    config.pitch_min_f0 = args.pitch_min_f0
    config.pitch_max_f0 = args.pitch_max_f0
    config.formant_number = args.formant_number
    config.mfcc_number = args.mfcc_number
    config.target_sample_rate = args.target_sample_rate
    return config


def flatten_result(
        result: Any,
        source_path: Path,
        speaker_type: str,
        label_source: str,
        label_notes: str = "",
) -> Dict[str, Any]:
    """
    Convert a FeatureExtractionResult into a flat dictionary for CSV export.

    This flattens nested structures (recording_quality, pitch, voice_quality,
    formants, MFCCs) into a single row of scalar values.
    """
    payload = {
        "session_id": result.session_id,
        "source_file": source_path.name,
        "source_path": str(source_path),
        "speaker_type": speaker_type,
        "label_source": label_source,
        "label_notes": label_notes,
        "processing_timestamp": result.processing_timestamp,
    }

    # Recording quality metrics
    rq = result.recording_quality
    payload.update(
        {
            "duration_seconds": rq.get("duration_seconds"),
            "snr_db": rq.get("snr_db"),
            "background_noise_level": rq.get("background_noise_level"),
            "sample_rate": rq.get("sample_rate"),
        }
    )

    # Pitch features
    pitch = result.features.get("pitch", {})
    payload.update(
        {
            "mean_f0_hz": pitch.get("mean_f0_hz"),
            "min_f0_hz": pitch.get("min_f0_hz"),
            "max_f0_hz": pitch.get("max_f0_hz"),
            "variability": pitch.get("variability"),
        }
    )

    # Voice quality features (jitter, shimmer, HNR)
    voice_quality = result.features.get("voice_quality", {})
    payload.update(
        {
            "jitter_percent": voice_quality.get("jitter_percent"),
            "shimmer_db": voice_quality.get("shimmer_db"),
            "harmonic_to_noise_ratio": voice_quality.get("harmonic_to_noise_ratio"),
        }
    )

    # Formants (F1, F2, F3, F4, ...)
    formants = result.features.get("timbre", {}).get("formants", {})
    for key, value in formants.items():
        payload[key] = value

    # MFCC coefficients
    mfccs = result.features.get("timbre", {}).get("mfccs", [])
    for idx, value in enumerate(mfccs, start=1):
        payload[f"mfcc_{idx}"] = value

    return payload


def pick_files(input_dir: Path, limit: Optional[int]) -> list[Path]:
    """Scan input directory and return list of supported audio files."""
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if limit is not None:
        return files[:limit]
    return files


def main() -> int:
    """
    Main entry point for batch voice dataset analysis.

    Workflow:
    1. Parse command-line arguments
    2. Load manifest (if provided) for speaker type filtering
    3. Configure feature extractor with custom parameters
    4. Iterate through audio files in input directory
    5. Extract features for each file (filtering by speaker type if manifest used)
    6. Write aggregated results to CSV and detailed JSON
    7. Report processing statistics (success, failures, skipped)

    Returns:
        0 on success, 1 on error.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    # Try to import project modules (will fail if dependencies not installed)
    try:
        from audio_processor import VoiceFeatureExtractor
        from config import DEFAULT_CONFIG
    except ModuleNotFoundError as exc:
        LOGGER.error(
            "Missing runtime dependency: %s. Install project requirements in a clean venv "
            "before running the extractor.",
            exc,
        )
        return 1

    # Validate input directory exists
    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return 1

    # Validate manifest requirement
    if args.speaker_type != "all" and args.manifest is None:
        LOGGER.error(
            "speaker-type=%s requires a manifest with filename,speaker_type. "
            "Without labels, we cannot reliably separate user and assistant audio.",
            args.speaker_type,
        )
        return 1

    # Load manifest if provided
    manifest = load_manifest(args.manifest) if args.manifest else {}

    # Create custom configuration from command-line arguments
    config = make_config(args, DEFAULT_CONFIG)
    extractor = VoiceFeatureExtractor(config)

    # Create output directory and prepare output files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]  # Unique identifier for this batch run
    csv_path = args.output_dir / f"voice_features_{run_id}.csv"
    json_path = args.output_dir / f"voice_features_{run_id}.json"

    processed_rows = []  # Successful extractions
    skipped_missing_label = 0  # Files without manifest label when filtering enabled
    failed_rows = []  # Files that raised exceptions

    # Scan input directory for audio files
    files = pick_files(args.input_dir, args.limit)
    LOGGER.info("Found %d candidate audio files", len(files))

    # Process each file
    for file_path in files:
        # Get label from manifest (if available)
        label = manifest.get(file_path.name)
        speaker_type = (label or {}).get("speaker_type", "unknown")
        label_source = "manifest" if label else "unlabeled"
        label_notes = (label or {}).get("notes", "")
        session_id = (label or {}).get("session_id") or str(uuid.uuid4())

        # Filter by speaker type if requested
        if args.speaker_type != "all":
            if not label:
                skipped_missing_label += 1
                continue
            if speaker_type != args.speaker_type:
                continue

        try:
            # Read audio file and extract features
            audio_data = file_path.read_bytes()
            result = extractor.extract_features(
                audio_data,
                session_id=session_id,
                original_filename=file_path.name,
            )

            # Skip files shorter than minimum duration
            if result.recording_quality.get("duration_seconds", 0.0) < args.min_duration:
                LOGGER.warning("Skipping short file: %s", file_path.name)
                continue

            # Flatten result and add to output
            processed_rows.append(
                flatten_result(
                    result,
                    source_path=file_path,
                    speaker_type=speaker_type,
                    label_source=label_source,
                    label_notes=label_notes,
                )
            )
            LOGGER.info("Processed %s", file_path.name)

        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"file": file_path.name, "error": str(exc)})
            LOGGER.exception("Failed to process %s", file_path.name)

    # Validate that we processed at least one file
    if not processed_rows:
        LOGGER.error("No files were processed successfully.")
        if skipped_missing_label:
            LOGGER.error(
                "%d files were skipped because they had no manifest label.",
                skipped_missing_label,
            )
        return 1

    # Write CSV with flattened features
    fieldnames = sorted({key for row in processed_rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    # Write detailed JSON with summary and full results
    summary = {
        "input_dir": str(args.input_dir),
        "manifest": str(args.manifest) if args.manifest else None,
        "speaker_type": args.speaker_type,
        "processed_count": len(processed_rows),
        "failed_count": len(failed_rows),
        "skipped_missing_label": skipped_missing_label,
        "output_csv": str(csv_path),
        "rows": processed_rows,
        "failures": failed_rows,
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Log final statistics
    LOGGER.info("Saved CSV: %s", csv_path)
    LOGGER.info("Saved JSON: %s", json_path)
    LOGGER.info("Processed %d files successfully", len(processed_rows))
    if failed_rows:
        LOGGER.info("%d files failed", len(failed_rows))
    if skipped_missing_label:
        LOGGER.info("%d files skipped because they are unlabeled", skipped_missing_label)

    return 0


if __name__ == "__main__":
    sys.exit(main())