# MP4 Vision API Reference

This is the implementation-oriented reference for `api/vision_api.py`. The endpoint returns Vision observations; it does not return a final risk score or alert decision.

## Endpoint and input

Start the server with the command in the root [README](../README.md#getting-started). The endpoint is `POST /analyze/vision` as `multipart/form-data`.

| Field | Type | Required | Notes |
| --- | --- | --- |
| `file` | MP4 file | yes | Only `.mp4`; server limit is 100 MiB. |
| `triggerId` | string | yes | Correlates the Vision result with the caller's trigger. |
| `triggerType` | `AUDIO` or `KEYPAD` | yes | Selects the Vision branch. |
| `securityEventId` | string | AUDIO: yes; KEYPAD: optional | Blank, `null`, and `none` are normalized to null before validation. |
| `triggeredAt` | ISO-8601 datetime | yes | Returned in the trigger object. |

The response has `status: "success"` and a `result` containing `trigger`, `analyzedAt`, `video`, `behavior`, `visionEvents`, and `observations`. `behavior` is `null` on `KEYPAD` and in the default rule-based `AUDIO` mode. `observations` records whether tracking/pose data was used; it is evidence, not a risk decision.

## Branches

### AUDIO (default: `exhibition_rules`)

Set `VISION_AUDIO_MODE=exhibition_rules`, or leave it unset. The endpoint applies:

1. OpenCV edge-density and Laplacian-variance rules for `CAMERA_OCCLUSION_SUSPECTED`, requiring a 0.3-second sustained low-structure interval.
2. YOLOv8n person detection with BoT-SORT tracking, followed by the five-second cumulative observation rule for `PERSON_DWELL_OBSERVED`.

Both event definitions use a window of at most the first 10 seconds. `VISION_EXHIBITION_LIMIT_TRACKING_TO_WINDOW=true` restricts the tracking pass to that already-used event window. `VISION_EXHIBITION_PARALLEL_ANALYSIS=true` runs the OpenCV and one tracking pass concurrently; it does not concurrently access the shared YOLO model.

### KEYPAD

The endpoint uses MediaPipe HandLandmarker for `SILENT_SIGNAL`, then YOLOv8n/BoT-SORT tracking for `REAR_CLOSE_APPROACH_SUSPECTED`. With `VISION_KEYPAD_PARALLEL_ANALYSIS=true`, the independent MediaPipe and tracking passes overlap; rear-approach evaluation occurs after tracking.

`VISION_ENABLE_WEAPON_DETECTION=true` enables the optional `WEAPON_PROXIMITY_SUSPECTED` rule. This is not enabled by default and is not part of the documented default event set.

`VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL=true` is also opt-in. When a supported Silent Signal configuration confirms early, the endpoint returns `analysisComplete: false` and `earlyReturnReason: "SILENT_SIGNAL"`; `trackedPersonCount` is `null`, because tracking and rear/weapon analysis were intentionally not run. This is a latency feature with an explicit incomplete-analysis contract, not an accuracy improvement.

### Preserved temporal classifier

`VISION_AUDIO_MODE=temporal_trial6` selects the retained `AUDIO` behavior-classification route. It additionally requires `VISION_MODEL` to be one of the allowed temporal choices. This route produces a behavior prediction and is preserved for experiment/backward-compatibility use; it is not the default exhibition pipeline. The model label `A18_20` is translated to API value `A18`, and `A20` remains a compatibility probability of `0.0` for this model.

## Errors and operational boundaries

- Invalid `VISION_AUDIO_MODE`, a missing AUDIO `securityEventId`, or incompatible operator-filter settings result in HTTP 422.
- Non-MP4 input is rejected with HTTP 415; oversize input is rejected with HTTP 413.
- Model files are loaded lazily. Missing required assets cause the request to fail rather than falling back to an unspecified model.
- A JSON `POST /triggers/keypad` endpoint only retains recent keypad trigger metadata in memory. It does not upload video or run Vision analysis.

The code is the source of truth for exact JSON fields and exceptional paths: [api/vision_api.py](../api/vision_api.py). Streaming requests, batching, callback retries, and polling are intentionally documented separately in the [Streaming Batch Frames API](../experiments/streaming_poc/streaming_api_poc/BATCH_API_SPEC.md).
