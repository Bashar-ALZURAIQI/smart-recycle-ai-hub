# YOLO26n V1 Training

## 1. Purpose

This document records the first official YOLO26n training baseline for the Smart Recycle AI-Hub V1 prototype.

The purpose of this model is to detect four waste categories:

- plastic
- metal
- glass
- paper_cardboard

Reject is not a YOLO training class.

Unknown, unsupported, or low-confidence objects will be handled by Python decision logic.

---

## 2. Model

Pretrained model:

```text
yolo26n.pt
```

The pretrained model was adapted from its original class configuration to the four waste classes used by Smart Recycle AI-Hub.

During training, Ultralytics reported:

```text
Overriding model.yaml nc=80 with nc=4
```

---

## 3. Training Command

The official V1 baseline was trained using:

```cmd
yolo detect train model=yolo26n.pt data=data\processed\waste_v1\data.yaml epochs=80 patience=15 imgsz=640 batch=16 device=0 workers=0 project=runs\waste_v1 name=yolo26n_v1_baseline
```

### Training Parameters

| Parameter | Value |
|---|---|
| Model | yolo26n.pt |
| Dataset config | data\processed\waste_v1\data.yaml |
| Epochs | 80 |
| Patience | 15 |
| Image size | 640 |
| Batch size | 16 |
| Device | 0 |
| Workers | 0 |
| Project | runs\waste_v1 |
| Run name | yolo26n_v1_baseline |

---

## 4. Training Environment

The training environment was:

```text
Python: 3.14.6
Ultralytics: 8.4.142
PyTorch: 2.14.0+cu126
CUDA: Enabled
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: approximately 6 GB
```

Automatic Mixed Precision (AMP) checks passed successfully.

---

## 5. Dataset Used

Training split:

```text
Images: 15,148
Objects: 83,850
```

Training class distribution:

```text
plastic: 18,785
metal: 13,018
glass: 18,480
paper_cardboard: 33,567
```

Validation split:

```text
Images: 1,268
Objects: 2,086
```

Validation class distribution:

```text
plastic: 497
metal: 298
glass: 456
paper_cardboard: 835
```

The dataset had already passed validation checks before training, including:

- invalid label checks
- invalid class ID checks
- bounding box checks
- empty label checks
- duplicate label checks
- train/validation source leakage checks

The final checks reported:

```text
EMPTY: 0
BAD_LINES: 0
BAD_CLASS: 0
BAD_BOX: 0
FILES WITH DUPLICATES: 0
TOTAL DUPLICATE LINES: 0
OVERLAPPING SOURCE GROUPS: 0
```

For full dataset preparation details, see:

```text
docs/DATASET_PREPARATION_V1.md
```

---

## 6. Training Completion

Training completed successfully:

```text
80 / 80 epochs
```

Total training time:

```text
24.163 hours
```

The final weights generated were:

```text
best.pt
last.pt
```

Each weight file was approximately:

```text
5.4 MB
```

The model weights are intentionally not tracked by Git because `.pt` files are excluded by `.gitignore`.

The training output was stored locally under the Ultralytics run directory.

---

## 7. Final Validation Results

After training, Ultralytics automatically validated the saved `best.pt` model.

The final overall results were:

| Metric | Result |
|---|---:|
| Precision | 0.809 |
| Recall | 0.747 |
| mAP50 | 0.802 |
| mAP50-95 | 0.452 |

---

## 8. Per-Class Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| plastic | 0.802 | 0.767 | 0.831 | 0.459 |
| metal | 0.877 | 0.861 | 0.902 | 0.498 |
| glass | 0.819 | 0.601 | 0.693 | 0.381 |
| paper_cardboard | 0.738 | 0.760 | 0.783 | 0.470 |

---

## 9. Current Model Assessment

Metal currently shows the strongest validation performance.

Its results were:

```text
Precision: 0.877
Recall: 0.861
mAP50: 0.902
mAP50-95: 0.498
```

Glass currently has the weakest recall:

```text
Recall: 0.601
```

Its complete results were:

```text
Precision: 0.819
Recall: 0.601
mAP50: 0.693
mAP50-95: 0.381
```

This means missed glass detections require special attention during real-world testing.

However, the model will not be retrained only because glass has lower validation performance.

Retraining will only be considered if real-world testing shows that the AI model itself is causing significant failures.

Possible non-AI causes must also be considered, including:

- lighting
- camera position
- camera quality
- object orientation
- conveyor positioning
- physical object appearance
- background conditions

---

## 10. Inference Speed

Final validation reported approximately:

```text
Preprocess: 0.2 ms/image
Inference: 6.2 ms/image
Postprocess: 0.8 ms/image
```

These measurements were produced during validation using the NVIDIA GeForce RTX 4050 Laptop GPU.

---

## 11. Reproducible Training Script

A Python version of the official V1 training configuration is stored at:

```text
scripts/train_yolo26n_v1.py
```

Its purpose is to preserve the exact baseline training configuration in source control.

The script should not be executed unless retraining is intentionally required.

The original training was executed using the Ultralytics CLI command documented in Section 3.

---

## 12. Git Tracking Policy

The following project assets are intentionally not committed to GitHub:

```text
*.pt
runs/
data/processed/
data/raw/
data/interim/
data/inspection/
```

This prevents large generated datasets, training outputs, and model weights from being stored directly in the Git repository.

The repository instead tracks:

- training scripts
- dataset preparation scripts
- project configuration
- AI source code
- automated tests
- documentation
- validation methodology

---

## 13. Training Milestone Status

The YOLO26n V1 baseline training milestone is complete.

Current training status:

```text
Dataset preparation: DONE
Dataset validation: DONE
Duplicate checks: DONE
Split leakage checks: DONE
YOLO26n training: DONE
80/80 epochs: DONE
best.pt generated: DONE
Final validation: DONE
```

---

## 14. Post-Training Software Integration

The first post-training software integration stage is now implemented.

The completed path is:

```text
best.pt
    ↓
Python inference layer
    ↓
real-world image testing
    ↓
Streamlit V1 image tester
    ↓
multiple-image upload
    ↓
image selection
    ↓
YOLO annotated result
    ↓
class + confidence table
    ↓
session statistics
    ↓
automated testing
```

The reusable inference implementation is stored in:

```text
ai/inference.py
```

The Streamlit testing interface is stored in:

```text
ai/streamlit_app.py
```

Detailed documentation is available at:

```text
docs/AI_INFERENCE_STREAMLIT_V1.md
```

---

## 15. Current Streamlit Capabilities

The current Streamlit V1 testing interface supports:

- multiple image upload
- JPG, JPEG, and PNG images
- image selection
- original image preview
- YOLO annotated image preview
- side-by-side comparison
- detected class display
- confidence display
- detection result table
- current image count
- unique images tested during the session
- total detection count during the session
- protection against duplicate counting caused by Streamlit reruns
- session reset

The Streamlit interface is a development and validation tool.

It is not intended to be the final physical-machine interface.

---

## 16. Real-World Testing Status

Real-world image testing has started.

The trained model has been tested through the Streamlit interface using real waste photographs.

Tests have produced detections for V1 classes including:

```text
plastic
metal
glass
```

The complete software path has therefore been exercised on real images:

```text
real waste image
    ↓
Streamlit
    ↓
image preparation
    ↓
best.pt
    ↓
YOLO inference
    ↓
bounding boxes
    ↓
class + confidence
    ↓
result display
```

---

## 17. Real-World Detection Observations

Real testing showed that the model may produce multiple detections in a single scene.

A scene may contain multiple valid objects, but the model can also generate lower-confidence detections.

For example, a test may return results similar to:

```text
plastic          0.755
plastic          0.580
plastic          0.339
paper_cardboard  0.294
```

This is important for the future physical sorting system.

The robotic sorting logic should not blindly convert every raw YOLO result into a hardware command.

A filtering and decision stage is required.

---

## 18. Blank / Simple Image Observation

During automated testing, a blank or simple artificial image could still occasionally produce a YOLO detection.

This indicates that the final sorting system requires application-level filtering.

This observation does not automatically justify retraining.

It will instead inform the confidence and decision logic implemented in the next development phase.

---

## 19. Automated Testing

Automated tests now cover the AI inference and Streamlit workflow.

Inference tests:

```text
tests/test_inference.py
```

Streamlit tests:

```text
tests/test_streamlit_app.py
```

The complete project test suite can be run using:

```cmd
python -m pytest -q
```

Current automated coverage includes:

- exact V1 class configuration
- model path resolution
- existence of `best.pt`
- trained model loading
- model class-name validation
- YOLO image inference
- detection extraction
- RGB image preparation
- Streamlit entry point
- multiple image upload
- image selection
- session statistics
- protection against duplicate counting
- total detection counting
- reset behavior
- side-by-side image layout
- detection table

---

## 20. Retraining Decision Rule

Retraining is not currently planned.

The current `best.pt` model will continue to be tested using real physical waste objects.

Retraining will only be considered if documented real-world tests show that model quality is a significant source of failure.

A problem caused by:

- lighting
- camera angle
- camera quality
- object position
- background
- conveyor mechanics
- sensors
- serial communication
- Arduino behavior
- robotic arm behavior

must not automatically be treated as a YOLO training problem.

---

## 21. Confidence Threshold

The current Streamlit implementation displays the detections returned by YOLO.

A final application confidence threshold has not yet been implemented.

The next phase will test a confidence threshold using real-world evidence.

A possible starting experimental value is:

```text
0.50
```

This value is not yet frozen.

The final threshold should be selected from real prototype testing rather than chosen only from one image.

---

## 22. Best-Detection and Decision Logic

The next software stage will convert raw detections into a sorting decision.

The planned software flow is:

```text
YOLO detections
    ↓
confidence threshold
    ↓
valid detections
    ↓
best / target detection
    ↓
class decision
    ↓
Reject or recyclable class
```

The exact decision rule will be implemented and tested before hardware commands are generated.

---

## 23. Reject Logic

Reject remains Python application logic.

Reject is not a fifth YOLO class.

The future decision logic may return Reject when:

```text
no acceptable detection exists
OR
confidence is below the accepted threshold
OR
the detection does not satisfy the V1 sorting rules
```

The exact Reject rule has not yet been finalized.

---

## 24. Next Development Phase

The immediate development path is now:

```text
Confidence threshold
    ↓
Best-detection selection
    ↓
Reject decision logic
    ↓
More real-world validation
    ↓
Live camera inference
    ↓
PySerial communication
    ↓
Arduino integration
    ↓
Conveyor + sensor + robotic arm
    ↓
First physical AI sorting test
    ↓
Integrated V1 prototype
```

Streamlit image testing is no longer a future task.

It is an implemented and tested component of the current V1 development workflow.

---

## 25. Live Camera Phase

The next major inference interface after image testing will be live camera inference.

The same reusable AI logic in:

```text
ai/inference.py
```

should be reused where practical rather than duplicating the YOLO inference implementation.

The live-camera stage will later need to consider:

- fixed camera placement
- lighting consistency
- pickup-zone visibility
- inference timing
- duplicate frame detections
- object tracking or decision timing

These are not implemented in the current image-testing phase.

---

## 26. Hardware Integration Direction

After the software decision logic is stable, the planned integration path is:

```text
Camera
  ↓
YOLO
  ↓
Python decision
  ↓
PySerial
  ↓
Arduino-compatible controller
  ↓
Conveyor / sensor / robotic arm
  ↓
Target bin
```

Hardware communication should only receive a final application decision rather than every raw YOLO detection.

---

## 27. V1 Classes

The V1 model is frozen to four YOLO classes:

```text
0: plastic
1: metal
2: glass
3: paper_cardboard
```

Reject is handled by application logic and is not a fifth YOLO class.

No additional waste categories should be added before the V1 prototype is successfully integrated and tested.

---

## 28. Related Documentation

Dataset preparation:

```text
docs/DATASET_PREPARATION_V1.md
```

Detailed inference and Streamlit documentation:

```text
docs/AI_INFERENCE_STREAMLIT_V1.md
```

AI module summary:

```text
ai/README.md
```

Project scope:

```text
docs/PROJECT_SCOPE_V1.md
```

Repository overview:

```text
README.md
```

---

## 29. Current Milestone Summary

Current project software status:

```text
Dataset preparation            DONE
Dataset validation             DONE
YOLO26n training               DONE
80/80 epochs                   DONE
best.pt generation             DONE
Final model validation         DONE
Reusable image inference       DONE
Detection extraction           DONE
Streamlit image tester         DONE
Multiple-image interface       DONE
Session statistics             DONE
Session reset                  DONE
Automated AI tests             DONE
Automated Streamlit tests      DONE
Real-world image testing       STARTED
Confidence decision logic      NEXT
Live camera inference          PENDING
Hardware integration           PENDING
```

The current priority is to convert tested YOLO detections into a reliable V1 sorting decision before moving to the physical machine.
