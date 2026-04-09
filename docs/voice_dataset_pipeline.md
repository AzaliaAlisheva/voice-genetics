# Dialogue Voice Pipeline

This project is moving from file-level voice analysis to conversation-level voice phenotyping.

## Target workflow

1. Receive a full phone conversation recording.
2. Run diarization to split the audio into speech turns.
3. Assign each turn a semantic role by matching the turn embedding against stable voice profiles: `assistant_1` is seeded from the first assistant voice, `user` from the first user voice, and any later distinct assistant voices become `assistant_2`, `assistant_3`, and so on. Hold music is labeled `music` only by a conservative fallback heuristic.
4. Extract acoustic features only from the user turns.
5. Standardize the result into numeric vectors for downstream genetic analysis.
6. Store derived features, not raw audio, in the analysis outputs.

## Why this matters

The project goal is to study how acoustic characteristics relate to genetic markers. For that use case, the important signal is the client's voice, not the full dialogue.

Raw audio should be treated as sensitive data:

- keep it outside git
- do not store it in feature output tables
- prefer secure backend storage or controlled object storage
- publish only derived vectors and metadata needed for analysis

### Storage layout

Backend storage is organized by stable `audio_id` values:

```text
storage/
  raw_audio/
    <audio_id>/
      source.mp3
      metadata.json
```

This layout is suitable for a local backend and maps cleanly to object storage concepts:

- `audio_id` is the external reference used by APIs
- `source.*` is the raw audio blob
- `metadata.json` keeps filename, checksum, content type, and lifecycle fields

## Manifest format

Use one row per dialogue segment. The minimum recommended columns are:

```csv
segment_id,start_sec,end_sec,analysis_start_sec,analysis_end_sec,role,speaker_id,speaker_cluster,turn_type,is_barge_in,text,source
turn_001,0.00,2.84,0.00,2.84,assistant_1,speaker_1,4,speech,false,"Hello, I need help",manifest
turn_002,2.90,4.20,3.15,4.20,user,speaker_2,5,barge_in,true,"Sure, let's continue",manifest
turn_003,4.25,5.30,4.25,5.30,assistant_2,speaker_3,2,speech,false,"Please confirm",manifest
```

Required fields:

- `segment_id`
- `start_sec`
- `end_sec`
- `role`

Allowed role values:

- `assistant_1`
- `user`
- `assistant_2`
- `assistant_3`
- `...`
- `music`

Optional fields:

- `speaker_id`
- `speaker_cluster`
- `analysis_start_sec`
- `analysis_end_sec`
- `turn_type`
- `is_barge_in`
- `text`
- `source`

## Two-step processing model

### 1. Diarization

Use `POST /dialogue/diarize` to split an audio file into speech turns and cluster the turns by speaker.

This produces:

- start and end times
- turn duration
- speaker cluster IDs
- a diarization summary

At this stage, the output is not yet user/assistant-aware.

If you need a downloadable manifest, use `POST /dialogue/manifest` to generate `dialogue_manifest.csv` directly from diarization output.

The generated manifest keeps the first turn as `assistant_1` and the second as `user`. After that, the turn roles continue numbering assistant voices when a new voice profile appears, and any hold music is labeled `music`. `speaker_id` is a stable canonical voice label, while `speaker_cluster` remains the technical clustering output. If barge-in is detected, `analysis_start_sec` trims the contaminated head of the segment before feature extraction.

### 2. Role-aware analysis

Use `POST /dialogue/analyze` with the audio file and the segment manifest.

This produces:

- user-only feature vectors
- per-segment acoustic features
- recording quality metadata
- speaker summaries

If a segment is not labeled with `role=user`, it is excluded from user-only aggregation.

## Local development

The repository can keep a local `data/` directory during development, but `data/` is ignored by git.

For a batch workflow, add a small CSV manifest next to the audio files and process the recordings through the dialogue endpoints or a future batch job.

## Production recommendation

The backend should own the raw audio lifecycle:

- upload from the client
- or receive audio once and store it behind an `audio_id`
- temporary or controlled persistent storage
- diarization
- role assignment
- feature extraction
- delete or expire raw audio according to policy

That keeps the repo clean and supports GDPR-style minimization: the analysis artifacts contain only what the downstream genetics workflow needs.
