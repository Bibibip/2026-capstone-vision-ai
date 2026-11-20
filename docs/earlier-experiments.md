# Earlier Behavior-Classification Experiments

This document preserves the behavior-classification work that previously occupied the root README. It is separate from the default exhibition-oriented Vision pipeline, which returns rule-based observation events for `AUDIO` and `KEYPAD` triggers.

## Initial block-level XGBoost pipeline

The initial experiment classified annotation blocks as `N1`, `A17`, `A18`, `A19`, `A20`, or `A21`.

| Label | Historical meaning |
| --- | --- |
| `N1` | Normal entrance activity: resident entry/exit, regular delivery, waiting, or leaving |
| `A17` | Repeated doorbell pressing |
| `A18` | Repeated attempts to open the door |
| `A19` | Repeated door kicking |
| `A20` | Attempting to look inside the door |
| `A21` | Prolonged or repeated door knocking |

```text
Video
→ YOLO person detection + BoT-SORT tracking
→ MediaPipe Pose extraction
→ Block-level behavior and pose features
→ XGBoost classification
→ N1 / A17–A21 prediction
```

The historical data split recorded in the prior README was:

| Split | Data used |
| --- | --- |
| Train | AIHub TL/TS, `mydata`, and augmentations derived from the original training data |
| Validation | Original AIHub VL/VS videos |
| Test | Separate exhibition-camera or held-out field footage, to be constructed later |

In that setup, `normal` and `delivery` from `mydata` were mapped to `N1`, and `lookingInside` to `A20`. Tracking and pose extraction were run per video; annotation `start_frame`/`end_frame` ranges were then used for block-level features. The historical workflow was:

```powershell
python .\utils\load_annotation.py
python .\utils\generate_annotations.py
python .\tracking\tracker.py
python .\tracking\extract_behavior_features.py
python .\tracking\pose_extractor.py
python .\tracking\merge_features.py
python .\models\train_XGBoost.py
```

`tracker.py` and `pose_extractor.py` were designed to skip videos with an existing per-video CSV, allowing this experimental pipeline to resume after interruption. Its documented outputs were:

| Path | Historical output |
| --- | --- |
| `splits/annotations/train_annotations.csv` | AIHub training-block manifest from TL/TS |
| `splits/annotations/valid_annotations.csv` | AIHub validation-block manifest from VL/VS |
| `splits/annotations/train_annotations_all.csv` | Combined AIHub, `mydata`, and augmentation manifest |
| `results/tracking/` | Per-video person-tracking results |
| `results/pose/` | Per-video pose features |
| `results/features/` | Merged block-level behavior and pose features |
| `results/models/xgboost_validation/` | Validation report, confusion matrix, and model files |

### Historical XGBoost result

The former README reported the following AIHub VL/VS hold-out validation result:

| Metric | Value |
| --- | ---: |
| Accuracy | 86.36% |
| Macro F1 | 68.82% |
| Weighted F1 | 86.20% |

This is a block-classification validation result from the initial experiment. It is not a result for `CAMERA_OCCLUSION_SUSPECTED`, `PERSON_DWELL_OBSERVED`, `SILENT_SIGNAL`, or `REAR_CLOSE_APPROACH_SUSPECTED`, and it is not a deployment-level Vision accuracy claim.

The original record also noted the large proportion of `N1`; Macro F1 was therefore reviewed alongside accuracy. Generalization to exhibition-camera or separate field footage was not established by this validation result.

## Field and temporal-model records

The repository also preserves XGBoost and temporal/GRU-family investigations. These records help document why classifier results should remain scoped to their own data and protocol.

- A 36-clip field evaluation of the existing XGBoost service path recorded accuracy 0.4167 and five-class macro F1 0.1866 after scoring `A18`/`A20` as merged `A18_20`. This was an evaluation of that classifier path, not of the final event rules.
- A post-hoc zoomed-out field exploration recorded `temporal_trial6` accuracy 0.3889 and five-class macro F1 0.2427. It was explicitly not used for model selection.
- The scale-augmentation fusion-GRU candidate process defined a fixed scene-key GroupKFold gate before any pilot decision. Passing the CV gate only made a candidate eligible for a newly recorded pilot; it did not itself select a serving model.

Candidate-comparison numbers use differing splits and repeated-seed protocols; they should not be ranked against one another without retaining those conditions.

## Preserved temporal fallback

The MP4 API retains a `temporal_trial6` route behind `VISION_AUDIO_MODE=temporal_trial6`. It uses YOLO tracking, sampled MediaPipe Pose features, and a temporal ONNX model. It is not selected by the default `exhibition_rules` mode.

An offline 36-clip Trial 6 evaluation recorded accuracy 80.56% and macro F1 0.7024 under independent 10-second-clip conditions. Longer uploads are not equivalent to those evaluation clips, even when the same model is selected.

## Scope boundary

The repository contains no code-backed basis here for claiming that the classifier experiment was replaced solely because one model was numerically superior, or that an unmeasured classifier metric transfers to the final camera environment. The current README therefore reports the final rule-based event path and links this page for the preserved classifier history.

## Repository artifacts

Source and augmented videos, generated manifests, run outputs, and model weights are excluded according to `.gitignore`; the repository does not claim that every experimental data artifact is versioned in Git.
