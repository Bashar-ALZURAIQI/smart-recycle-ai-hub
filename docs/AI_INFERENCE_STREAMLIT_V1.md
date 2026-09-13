# AI Inference and Streamlit V1

## 1. Purpose

This document records the first complete image-inference and Streamlit testing workflow for Smart Recycle AI-Hub V1.

The goal of this stage was to move from a trained YOLO26n model to a reusable software inference layer and a practical local interface for testing real-world waste images.

This stage does not control the conveyor, sensor, Arduino, or robotic arm.

It is the software validation stage before live camera and hardware integration.

---

## 2. Starting Point

Before this phase began, the following milestones were already complete:

```text
Dataset preparation: DONE
Dataset validation: DONE
Duplicate-label checks: DONE
Train/validation leakage checks: DONE
YOLO26n training: DONE
80/80 epochs: DONE
best.pt generated: DONE
Final YOLO validation: DONE
```

The trained V1 model contains four classes:

```text
0: plastic
1: metal
2: glass
3: paper_cardboard
```

`Reject` is not a YOLO class.

Reject will be handled later by Python application logic.

---

## 3. Trained Model Location

The current V1 inference code expects the trained model locally at:

```text
runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt
```

The model weight file is intentionally not committed to GitHub.

The project `.gitignore` excludes:

```text
*.pt
runs/
```

A collaborator must therefore obtain `best.pt` separately before running local inference.

---

## 4. Streamlit Dependency

Streamlit was added to the project environment and recorded in:

```text
requirements.txt
```

Current recorded version:

```text
streamlit==1.63.0
```

The installed version can be checked using:

```cmd
python -m streamlit --version
```

---

## 5. AI Inference Module

Reusable inference logic is stored in:

```text
ai/inference.py
```

The module was deliberately separated from the Streamlit interface so the same AI logic can later be reused by:

- live camera inference
- decision logic
- conveyor integration
- hardware control
- automated tests

---

## 6. Supported Classes

The inference module defines the frozen V1 classes:

```python
SUPPORTED_CLASSES = \[
    "plastic",
    "metal",
    "glass",
    "paper_cardboard",
]
```

Automated tests verify that the trained model contains these same class names.

This protects the V1 system from accidentally loading a model trained with a different class configuration.

---

## 7. Project Root and Model Path

The model path is resolved relative to the repository instead of using a hard-coded computer-specific path.

The code determines the repository root from the location of:

```text
ai/inference.py
```

Then constructs the expected model location under:

```text
runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt
```

This avoids permanently tying the source code to one Windows username or drive path.

---

## 8. Model Loading

The inference module provides a model-loading function.

Its responsibility is to load the official trained V1 `best.pt` model using Ultralytics YOLO.

The model is loaded for inference only.

The Streamlit workflow does not call:

```python
model.train(...)
```

Therefore, running the image-testing application does not retrain or modify the model.

---

## 9. Image Inference

The reusable inference function accepts:

```text
model
image
```

and executes YOLO prediction.

The general flow is:

```text
Input image
    ↓
YOLO26n best.pt
    ↓
model.predict(...)
    ↓
Ultralytics result
```

The prediction function is configured to avoid unnecessary verbose terminal output during normal interface use.

---

## 10. Detection Extraction

YOLO results are converted into a simpler application-friendly structure.

Each detected object is represented using:

```text
class
confidence
```

Example:

```python
{
    "class": "plastic",
    "confidence": 0.91,
}
```

This format is easier to use later for:

- Streamlit tables
- decision logic
- Reject logic
- Arduino commands
- logging
- testing

---

## 11. Annotated Image Generation

Ultralytics provides an annotated image through the detection result.

The current application uses the YOLO result to generate an image containing:

- bounding boxes
- class labels
- confidence values

The generated image is converted to the correct color-channel order before display in Streamlit.

---

## 12. Streamlit Application

The V1 testing interface is stored at:

```text
ai/streamlit_app.py
```

The application is intended as an AI development and validation tool.

It is not the final interface for the physical recycling machine.

---

## 13. Running the Application

From the repository root, activate the virtual environment and run:

```cmd
python -m streamlit run ai\\streamlit_app.py
```

Streamlit normally starts the application locally at:

```text
http://localhost:8501
```

The server can be stopped from the terminal using:

```text
Ctrl + C
```

---

## 14. Image Preparation

Uploaded image files are opened using Pillow.

Every uploaded image is converted to:

```text
RGB
```

before inference.

This allows images such as transparent PNG files using RGBA mode to be normalized into a format suitable for the current inference workflow.

Supported file formats in the current interface are:

```text
JPG
JPEG
PNG
```

---

## 15. Multiple Image Upload

The original Streamlit prototype accepted only one image.

The V1 tester was expanded to support:

```text
accept_multiple_files=True
```

The user can therefore upload multiple real waste images in one testing session.

---

## 16. Image Selection

Uploading many images does not automatically run every image simultaneously.

Instead, uploaded file names are displayed through:

```text
Select image
```

The user selects which uploaded image to inspect.

This avoids unnecessarily running inference on every uploaded image whenever the Streamlit application reruns.

The selected image is then:

```text
prepared
→ analyzed
→ displayed
```

---

## 17. User Interface Layout

The current application uses a wide Streamlit page layout.

The original and YOLO-annotated images are displayed side by side:

```text
┌─────────────────────────┬─────────────────────────┐
│ Original image          │ YOLO detection result   │
│                         │                         │
│       IMAGE             │       IMAGE             │
│                         │ boxes + labels          │
└─────────────────────────┴─────────────────────────┘
```

This replaced the earlier design where the two images were displayed vertically and appeared excessively large.

---

## 18. Detection Results Table

Below the image comparison, the application displays a detection table.

Current columns:

```text
Class
Confidence
```

Example:

| Class | Confidence |
|---|---:|
| plastic | 0.755 |
| metal | 0.580 |
| glass | 0.421 |

Confidence values are rounded for readability in the interface.

---

## 19. Session Metrics

The application currently displays three metrics:

```text
Current images
Tested this session
Total detections
```

They are displayed side by side.

---

## 20. Current Images

`Current images` represents the number of images currently present in the uploader.

Example:

```text
3 files uploaded
→ Current images = 3
```

If the upload widget is cleared or reset:

```text
Current images = 0
```

---

## 21. Tested This Session

Streamlit automatically reruns the application when widgets change.

Without protection, the same image could therefore be counted repeatedly.

To prevent this, every tested image receives a content-based SHA-256 identifier.

Conceptually:

```text
image bytes
    ↓
SHA-256
    ↓
unique image ID
```

The current session stores tested image identifiers in:

```text
st.session_state
```

The same image content is therefore counted only once during the session.

Example:

```text
Analyze image A
→ Tested this session = 1
Streamlit reruns image A
→ Tested this session = 1
Select image B
→ Tested this session = 2
```

---

## 22. Total Detections

The application also stores the detection count for each unique tested image.

Example:

```text
Image A → 4 detections
Image B → 2 detections
Image C → 3 detections
```

The session total becomes:

```text
Total detections = 9
```

The same image is not added repeatedly during Streamlit reruns.

---

## 23. Session State

The current application uses Streamlit session state for:

```text
tested_image_ids
detection_counts
uploader_version
```

### tested_image_ids

Tracks unique image contents already analyzed during the current session.

### detection_counts

Stores the number of detections associated with each tested image.

### uploader_version

Allows the application to create a fresh uploader widget after a session reset.

---

## 24. Reset Session

The interface contains:

```text
Reset session
```

The reset operation:

```text
clears tested image history
clears detection counts
creates a fresh uploader
returns Current images to 0
returns Tested this session to 0
returns Total detections to 0
```

Resetting does not:

```text
delete files from the computer
modify best.pt
retrain YOLO
modify the dataset
```

---

## 25. Real-World Image Testing

The Streamlit interface has been tested manually using real waste photographs.

Tests included scenes containing multiple recyclable objects.

The current trained model successfully produced detections for V1 categories including:

```text
plastic
metal
glass
```

This confirmed that the complete software path works on real uploaded images:

```text
real image
    ↓
Streamlit upload
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
Streamlit result display
```

---

## 26. Real-World Observation: Multiple Detections

Real testing also showed that the model may return multiple detections in one scene.

For example, a single waste image may generate:

```text
plastic 0.755
plastic 0.580
plastic 0.339
paper_cardboard 0.294
```

This does not automatically mean the model must be retrained.

The next application stage will determine which detections should be accepted by the sorting decision logic.

---

## 27. Real-World Observation: False Positives

During automated inference testing, even an artificial blank or simple image could occasionally produce a detection.

This is useful evidence that the final machine should not blindly accept every raw YOLO result.

The V1 decision system will therefore require filtering and decision rules before hardware commands are generated.

---

## 28. Confidence Threshold

A final confidence threshold has not yet been implemented.

The next stage will introduce a threshold to suppress low-confidence detections.

A possible initial experimental value is:

```text
0.50
```

However, the final threshold should be selected from real-world test evidence rather than chosen only to make individual examples look correct.

---

## 29. Best Detection Logic

The current Streamlit application displays all returned detections.

The future physical sorting workflow will require a clearer decision.

For the V1 prototype, a likely software path is:

```text
YOLO detections
      ↓
confidence filtering
      ↓
valid detections
      ↓
best / target detection
      ↓
class decision
      ↓
sorting command
```

This logic has not yet been finalized.

---

## 30. Reject Logic

Reject remains application logic and is not part of the YOLO class list.

A future V1 decision may route an object to Reject when:

- no acceptable detection exists
- confidence is below the accepted threshold
- the detected result does not satisfy the sorting decision rules

The exact Reject rule will be implemented and tested in the next software stage.

---

## 31. Automated Testing Strategy

Development followed a test-first workflow.

New behavior was generally introduced using:

```text
RED
→ failing automated test
GREEN
→ minimal implementation
VERIFY
→ full test suite
```

This approach was used for the inference layer and Streamlit functionality.

---

## 32. Inference Tests

Inference tests are stored in:

```text
tests/test_inference.py
```

They currently verify behavior including:

- exact V1 supported classes
- correct model path
- existence of `best.pt`
- loading the trained model
- matching trained model class names
- executing inference
- returning an Ultralytics result
- extracting structured detections

---

## 33. Streamlit Tests

Streamlit tests are stored in:

```text
tests/test_streamlit_app.py
```

They currently verify behavior including:

- application entry point
- RGB image preparation
- multiple-file uploader
- image selector
- multiple image selection
- current image count
- unique tested-image count
- protection against duplicate rerun counting
- multiple unique images counted correctly
- total detection counting
- total detections not duplicated after rerun
- reset button existence
- reset behavior
- side-by-side session metrics
- original and annotated image display
- side-by-side image layout
- detection table
- end-to-end image analysis behavior

---

## 34. Full Test Command

Run all automated project tests with:

```cmd
python -m pytest -q
```

This is the preferred verification command before committing AI or Streamlit changes.

---

## 35. Git Policy

The following source files are intended to be tracked:

```text
ai/inference.py
ai/streamlit_app.py
tests/test_inference.py
tests/test_streamlit_app.py
requirements.txt
documentation files
```

The following generated assets are intentionally excluded:

```text
*.pt
runs/
data/raw/
data/processed/
data/interim/
data/inspection/
```

---

## 36. Current V1 Software Status

Current status after this phase:

```text
Dataset preparation            DONE
Dataset validation             DONE
YOLO26n training               DONE
best.pt generation             DONE
YOLO validation                DONE
Reusable image inference       DONE
Detection extraction           DONE
Streamlit image tester         DONE
Multiple image upload          DONE
Image selection                DONE
Session metrics                DONE
Session reset                  DONE
Automated inference tests      DONE
Automated Streamlit tests      DONE
Real-world image testing       STARTED
```

---

## 37. Not Yet Implemented

The following items are intentionally outside the current completed phase:

```text
final confidence threshold
best-detection selection
Reject decision logic
live camera inference
object pickup-zone tracking
PySerial integration
Arduino commands
conveyor control
sensor integration
robotic arm control
physical sorting
```

---

## 38. Immediate Next Steps

The recommended next development sequence is:

```text
1\. Confidence threshold
        ↓
2\. Best-detection selection
        ↓
3\. Reject decision logic
        ↓
4\. More real-world validation
        ↓
5\. Live camera inference
        ↓
6\. PySerial integration
        ↓
7\. Arduino + conveyor + sensor
        ↓
8\. Robotic arm command
        ↓
9\. First physical sorting test
```

The software should remain focused on V1 integration rather than adding new classes or unrelated features.

---

## 39. Retraining Rule

Retraining is not automatically required because of:

- lower-confidence detections
- duplicate detections
- a single false positive
- one difficult photograph

Before retraining, the project must determine whether failures are caused by:

- model quality
- lighting
- camera angle
- object position
- object appearance
- background
- image quality
- mechanical conditions

Retraining should only occur when documented testing shows that model performance is a significant cause of prototype failure.

---

## 40. Related Documentation

Dataset preparation:

```text
docs/DATASET_PREPARATION_V1.md
```

YOLO26n training:

```text
docs/YOLO26N_TRAINING_V1.md
```

Main project scope:

```text
docs/PROJECT_SCOPE_V1.md
```

AI module summary:

```text
ai/README.md
```

Repository overview:

```text
README.md
```

---

## 41. V1 Scope Rule

The V1 YOLO class list remains frozen:

```text
plastic
metal
glass
paper_cardboard
```

Reject is handled outside YOLO.

No new waste categories should be added until the integrated V1 prototype has been completed and tested.
