# Entrance Behavior Vision Pipeline

A Vision pipeline that classifies normal activities and anomalous behaviors in entrance camera footage on a block-by-block basis.

## Classification Labels

| Label | Meaning |
|---|---|
| N1 | Normal entrance activity: resident entry/exit, regular delivery, waiting, or leaving |
| A17 | Repeatedly pressing the doorbell |
| A18 | Repeated attempts to open the door |
| A19 | Repeatedly kicking the door |
| A20 | Attempting to look inside the door |
| A21 | Prolonged or repeated knocking on the door |

## Data Split

| Split | Data Used |
|---|---|
| Train | AIHub TL/TS, mydata, and augmented data derived from the original training data |
| Validation | Original AIHub VL/VS videos only |
| Test | Separate exhibition-camera footage or held-out field footage, to be constructed later |

`normal` and `delivery` from mydata are used as N1, while `lookingInside` is used as A20.

## Processing Pipeline

```text
Video
→ YOLO person detection + BoT-SORT tracking
→ MediaPipe Pose extraction
→ Aggregation of behavior and pose features by annotation block
→ XGBoost classification
→ N1 / A17–A21 predictions
```

- Tracking and pose extraction run only once per video.
- Feature extraction and training use the `start_frame` and `end_frame` ranges of each annotation block.
- Augmented training data uses the original block boundaries.

## Execution Order

Activate the virtual environment and run the following commands from the project root.

```powershell
python .\utils\load_annotation.py
python .\utils\generate_annotations.py
python .\tracking\tracker.py
python .\tracking\extract_behavior_features.py
python .\tracking\pose_extractor.py
python .\tracking\merge_features.py
python .\models\train_XGBoost.py
```

`tracker.py` and `pose_extractor.py` skip processing when per-video CSV files already exist, allowing the pipeline to be safely re-run after interruption.

## Generated Outputs

| Path | Contents |
|---|---|
| `splits/annotations/train_annotations.csv` | AIHub training-block manifest based on TL/TS |
| `splits/annotations/valid_annotations.csv` | Validation-block manifest based on VL/VS |
| `splits/annotations/train_annotations_all.csv` | Combined training manifest including AIHub, mydata, and augmentations |
| `results/tracking/` | Per-video person-tracking results |
| `results/pose/` | Per-video frame-level pose features |
| `results/features/` | Merged block-level behavior and pose features |
| `results/models/xgboost_validation/` | Validation report, confusion matrix, and model files |

## Current Baseline Results

AIHub VL/VS hold-out validation results:

* Accuracy: **86.36%**
* Macro F1-score: **68.82%**
* Weighted F1-score: **86.20%**

Because N1 accounts for a large proportion of the data, Macro F1-score is evaluated alongside Accuracy rather than relying on Accuracy alone. These results represent validation performance; final generalization performance should be further evaluated using exhibition-camera footage or separate field-test videos.

## Exhibition Vision FastAPI

`POST /analyze/vision` accepts an MP4 file and trigger metadata as `multipart/form-data`. Vision returns observation events or existing behavior-classification results only; LOW/MEDIUM/HIGH risk decisions are the responsibility of the Agent.

| `triggerType` | Default processing |
|---|---|
| `AUDIO` | `exhibition_rules`: accumulated person-observation and camera-occlusion rules |
| `KEYPAD` | Existing `SILENT_SIGNAL` and `REAR_CLOSE_APPROACH_SUSPECTED` rules |

`VISION_AUDIO_MODE` applies only to the AUDIO path. When unset or set to `exhibition_rules`, it uses the exhibition rules; setting it to `temporal_trial6` runs the preserved Temporal Trial 6 behavior-classification model. The KEYPAD path retains its existing rules regardless of this setting.

```powershell
# Default exhibition mode (the same behavior as an unset VISION_AUDIO_MODE)
$env:VISION_AUDIO_MODE = "exhibition_rules"
.\.venv\Scripts\python.exe .\api\vision_api.py

# Legacy Trial 6 fallback mode
$env:VISION_AUDIO_MODE = "temporal_trial6"
$env:VISION_MODEL = "temporal_trial6"
.\.venv\Scripts\python.exe .\api\vision_api.py
```

In exhibition-rules mode, AUDIO responses set `behavior` to `null` and return the following observation events in `visionEvents`.

| Event | Detection summary |
|---|---|
| `PERSON_DWELL_OBSERVED` | The same BoT-SORT track ID is observed for a cumulative five seconds or longer |
| `CAMERA_OCCLUSION_SUSPECTED` | Low edge density and low Laplacian variance persist for at least 0.3 seconds during the first 10 seconds |

The `observations` object in this mode includes the optional field `analysisMode: "exhibition_rules"`. This field is not added to `temporal_trial6` or existing KEYPAD responses. The `api/vision_api.py` implementation is the source of truth for the detailed API request and response contract.

## Streaming Vision API

The existing MP4 API (port 8000) remains available as a fallback. The Streaming
API (port 8010) sends JPEG frames from a single seven-second Raspberry Pi
capture directly to Vision while MP4 recording proceeds in parallel through
the Spring Boot/S3 storage path.

```text
Raspberry Pi single capture
  ├─ JPEG → Streaming Vision :8010 → EARLY_EVENT callback / FINAL_RESULT polling
  └─ MP4 → Spring Boot → S3

Streaming failure or partial result → existing MP4 `/analyze/vision` fallback
```

```powershell
# Existing MP4 Vision API (fallback)
$env:VISION_AUDIO_MODE = "exhibition_rules"
$env:SILENT_SIGNAL_COORDINATE_MODE = "front"
$env:REAR_APPROACH_CAMERA_MODE = "front"
.\.venv\Scripts\python.exe -m uvicorn api.vision_api:app --host 0.0.0.0 --port 8000

# Streaming Vision API (can run alongside the fallback API on the same machine)
.\.venv\Scripts\python.exe -m uvicorn experiments.streaming_poc.streaming_api_poc.app:app --host 0.0.0.0 --port 8010

# Or run the startup script with its operational environment variables from the project root
.\start-vision.ps1
```

The Streaming API exposes `POST /vision/start`, `POST /vision/frames` (recommended;
atomic batch upload of up to five JPEG frames), the compatibility endpoint
`POST /vision/frame`, `POST /vision/complete`, `GET /vision/status/{sessionId}`,
and `GET /vision/result/{sessionId}`. The Pi associates a `sessionId` with the
existing `triggerId`; Silent Signal is recorded separately through an
`EARLY_EVENT` callback during capture. Spring Boot polls the final result and
uses it exactly once for Agent reasoning for the same `triggerId`.

Batch requests validate JPEG bytes with SHA-256 and safely handle retransmission
of the same `batchId`, metadata, and bytes as a duplicate. The Pi must retain
the original JPEGs in a disk spool until it receives a success response, and
must retry the same batch after an ordering conflict (409) or queue saturation
(429). See the [Batch Frames API specification](experiments/streaming_poc/streaming_api_poc/BATCH_API_SPEC.md)
for the complete multipart contract and failure-handling rules.

The representative seven-second localhost validation configuration is KEYPAD
at `960x540`, JPEG Q85, 30 FPS (Silent Signal/Rear Approach), and AUDIO at
`1920x1080`, JPEG Q85, 30 FPS (Camera Occlusion/Person Dwell). Because 15/20
FPS produced `partial` results, those rates are not approved for final analysis
operation. Actual Raspberry Pi throughput for 210 frames in seven seconds and
Spring Boot authentication/network integration remain to be verified.

## Git Management

Source and augmented videos, run results, generated manifests, and model weights are excluded through `.gitignore`. Git tracks only source code and the configuration files needed for reproducibility.
