# AI Module

This folder contains the artificial intelligence and computer vision components of Smart Recycle AI-Hub.

The current V1 AI workflow is based on a trained YOLO26n object detection model and a Streamlit interface for real-world image testing.

---

## V1 AI Goal

The V1 AI system detects four recyclable waste categories:

```text
0: plastic
1: metal
2: glass
3: paper_cardboard
```

`Reject` is not a YOLO class.

Unknown objects, unsupported objects, or detections below the future decision threshold will be handled by Python decision logic.

---

## Current AI Stack

The current V1 AI stack includes:

- Python
- Ultralytics
- YOLO26n
- PyTorch
- OpenCV
- Pillow
- Streamlit
- Pytest

---

## Current Files

### `inference.py`

This module contains the reusable AI inference logic.

Current responsibilities:

- define the four V1 supported classes
- resolve the local trained model path
- load the trained `best.pt` model
- run YOLO inference on an image
- extract class names and confidence scores from YOLO results

The inference code is intentionally separated from Streamlit so that the same AI logic can later be reused by:

- live camera inference
- conveyor decision logic
- hardware integration
- automated testing

---

### `streamlit_app.py`

This module provides the current V1 image-testing interface.

Current features:

- multiple image upload
- JPG, JPEG, and PNG support
- image selection through a dropdown
- original image preview
- YOLO annotated image preview
- side-by-side image layout
- detected class display
- confidence score display
- detection result table
- current uploaded image count
- unique images tested during the current session
- total detections during the current session
- protection against duplicate counting after Streamlit reruns
- session reset button

The Streamlit application is a development and validation tool.

It is not intended to be the final physical-machine interface.

---

## Trained Model

The current V1 model is the official trained YOLO26n baseline.

Expected local model location:

```text
runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt
```

The model file is intentionally excluded from GitHub.

The repository `.gitignore` excludes:

```text
*.pt
runs/
```

A collaborator who wants to run the AI locally must obtain the trained model separately and place it in the expected path.

---

## Model Classes

The model names are expected to match exactly:

```text
plastic
metal
glass
paper_cardboard
```

The automated tests verify that the loaded trained model contains these four classes.

---

## Running the Streamlit Interface

From the repository root, with the virtual environment activated:

```cmd
python -m streamlit run ai\\streamlit_app.py
```

The interface normally becomes available locally at:

```text
http://localhost:8501
```

The user can upload multiple waste images and select which image to analyze.

---

## Current Streamlit Flow

The current image-testing flow is:

```text
Uploaded image
      ↓
Pillow image preparation
      ↓
RGB conversion
      ↓
Load best.pt
      ↓
YOLO26n inference
      ↓
YOLO result
      ↓
Annotated image
      ↓
Detection extraction
      ↓
Class + confidence
      ↓
Streamlit result table
```

---

## Session Statistics

The application currently displays three session metrics:

```text
Current images
Tested this session
Total detections
```

### Current images

Counts the number of files currently present in the uploader.

If images are removed, this number decreases.

### Tested this session

Counts unique image contents that have actually been analyzed during the current Streamlit session.

The same image is not counted repeatedly because Streamlit reruns the application.

A SHA-256 hash of the image contents is used as the unique image identifier.

### Total detections

Stores the number of YOLO detections generated for each unique tested image.

The same image is not added repeatedly during Streamlit reruns.

---

## Reset Session

The interface contains:

```text
Reset session
```

Resetting the session:

- clears tested-image history
- clears stored detection counts
- resets the uploaded image widget
- returns session statistics to zero

It does not delete any image from the user's computer.

It does not modify the trained model.

---

## Automated Tests

AI and interface tests are stored in:

```text
tests/test_inference.py
tests/test_streamlit_app.py
```

Run the complete project test suite with:

```cmd
python -m pytest -q
```

The current tests cover:

- V1 class configuration
- trained model path
- model loading
- YOLO image inference
- detection extraction
- image RGB preparation
- Streamlit main function
- multiple image upload
- image selection
- multiple image selector behavior
- current image count
- unique session image count
- duplicate rerun protection
- total detection count
- reset session behavior
- side-by-side image layout
- detection result table

---

## Real-World Testing Observations

The Streamlit interface has already been used with real waste images.

The trained model successfully produced detections across the V1 classes, including examples containing:

- plastic
- metal
- glass

Real-world tests also showed that YOLO can return multiple detections for the same scene, including lower-confidence detections.

This is expected to be handled in the next software stage rather than immediately retraining the model.

---

## Current Limitation

The current inference pipeline displays all YOLO detections returned by the model.

The application does not yet apply the final V1 decision logic.

The following features are not yet implemented:

- configurable confidence threshold
- final best-detection selection
- Reject decision logic
- live camera inference
- conveyor object tracking
- PySerial communication
- Arduino commands
- robotic arm control

---

## Next AI Development Steps

The immediate AI path is:

```text
Current image inference
        ↓
Confidence threshold
        ↓
Best detection selection
        ↓
Reject decision logic
        ↓
Live camera inference
        ↓
Hardware communication
```

The model should not be retrained only because lower-confidence or duplicate detections appear.

Retraining should be considered only after documented real-world testing shows that model quality itself is a significant source of failure.

---

## Training Documentation

For the official V1 YOLO26n training details, see:

```text
docs/YOLO26N_TRAINING_V1.md
```

For detailed Streamlit and inference implementation notes, see:

```text
docs/AI_INFERENCE_STREAMLIT_V1.md
```

For dataset preparation details, see:

```text
docs/DATASET_PREPARATION_V1.md
```

---

## V1 Scope Rule

The four detection classes are currently frozen.

Do not add new waste classes before the integrated V1 prototype has been completed and tested.
