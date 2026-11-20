# Evaluation Conditions and Limitations

This note separates measurements by interface and experiment. None of the values below is a system-wide latency SLA or an accuracy score for all Vision events.

## MP4 API optimization validation

The representative HTTP comparison used fresh local API processes, Windows 11, Python 3.14.3, Torch 2.13.0+cu130, and an RTX 4060 (8 GiB). The default rule-based `AUDIO` mode was used.

| Request | Default HTTP wall time | Relevant opt-in flags enabled | Result |
| --- | ---: | ---: | --- |
| Camera-occlusion clip / `AUDIO` | 39.235 s | 21.856 s | Same recorded schema, event payloads, and two tracked persons in this pair. |
| Approach-signal clip / `KEYPAD` | 48.417 s | 28.711 s | Same recorded response pair, but no events in the new baseline/optimized pair. |

These times contain the client-visible HTTP request duration. The endpoint has no independent timer for multipart storage, decoding alone, event merge, or JSON serialization. For the measured clips, the main recorded stages were YOLO/BoT-SORT and, for KEYPAD, HandLandmarker; OpenCV/YOLO or HandLandmarker/YOLO overlap explains most of the reduced wall time. See the full [optimization report](../experiment_results/vision_api_optimization/REPORT.md).

The KEYPAD pair cannot establish event-regression clearance: an older stored run of the same clip contains Silent Signal and rear-approach events that the new baseline did not reproduce. The report treats that as a reproducibility issue to resolve before a broad parity replay.

## Streaming measurements

The independent streaming server was smoke-tested on localhost. In one Phase D KEYPAD replay, 210 contiguous 960×540 JPEG frames (Q85) were sent in 6.994 seconds. The final result was available 8.357 seconds after capture start, with a 1.364-second post-complete drain. It produced Silent Signal at frame 114, rear approach at frame 43, and three tracked persons. This is a local capture-to-final-result measurement, not a Raspberry Pi delivery, Spring Boot, S3, MQTT, callback-authentication, or Agent latency measurement. [Phase D details](../experiment_results/streaming_poc/streaming_api_poc/phase_d/PHASE_D_REPORT.md).

## Evidence-backed limitations

- In the retained default Silent Signal rule, two human-reviewed negative videos generated early candidates. Making the signal available early also makes those baseline false positives available early; it does not validate the rule’s accuracy.
- The tested camera-occlusion threshold is resolution-sensitive: the representative positive event was absent at 960×540 with unchanged thresholds. Original-resolution 30-FPS JPEG reproduced the listed occlusion frames in the limited test.
- For the tested rear-approach scenario, 10-FPS online input did not detect the event. A 15-FPS result changed event timing/details and was not declared regression-equivalent.
- 30-FPS JPEG processing accumulated a local YOLO backlog in representative replays. Actual 210-frame/7-second Raspberry Pi delivery remains unmeasured.
- Broader positive/negative event regression, production callback security, persistent session storage, abandoned-session cleanup, multi-session isolation, and production network integration remain unverified. [Phase D report](../experiment_results/streaming_poc/streaming_api_poc/phase_d/PHASE_D_REPORT.md)

## What is intentionally not concluded

- No rule-event precision, recall, or aggregate detection accuracy is claimed from the representative replays.
- Offline classifier metrics are not evidence of the default rule-based final system’s event-detection accuracy.
- A locally overlapped capture/analysis path does not demonstrate lower model inference cost or an end-to-end deployment latency improvement.
