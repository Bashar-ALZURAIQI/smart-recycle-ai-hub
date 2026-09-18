# ♻️ Smart Recycle AI-Hub

### AI-Powered Waste Detection and Robotic Sorting Prototype

Smart Recycle AI-Hub is a university-scale smart recycling prototype that combines **artificial intelligence, computer vision, Python, YOLO, sensors, conveyor control, and robotic sorting**.

The V1 system uses a fixed camera and a trained **YOLO26n** model to detect recyclable waste. The next stages connect the AI decision layer with an Arduino-compatible controller, conveyor system, sensors, and a robotic arm to sort objects into the correct recycling bins.

---

## 🚀 Project Status

**Current Stage:** AI detection and software inference completed — physical system integration in progress.

| Component                   | Status         |
| --------------------------- | -------------- |
| Dataset preparation         | ✅ Completed    |
| Dataset validation          | ✅ Completed    |
| YOLO26n training            | ✅ Completed    |
| Model validation            | ✅ Completed    |
| Python inference            | ✅ Completed    |
| Streamlit testing interface | ✅ Completed    |
| Automated tests             | ✅ Completed    |
| Decision & reject logic     | 🔄 In Progress |
| Live camera inference       | ⏳ Planned      |
| Arduino communication       | ⏳ Planned      |
| Conveyor integration        | ⏳ Planned      |
| Robotic arm integration     | ⏳ Planned      |
| Complete V1 prototype       | ⏳ Planned      |

---

## 🧠 AI Model Performance

The first official YOLO26n V1 baseline was trained on the project's validated recycling dataset.

| Metric    |    Result |
| --------- | --------: |
| Precision | **0.809** |
| Recall    | **0.747** |
| mAP50     | **0.802** |
| mAP50-95  | **0.452** |

### Dataset

* **15,148** training images
* **83,850** training objects
* **1,268** validation images
* **2,086** validation objects

---

## 🛠️ Core Technologies

`Python` · `YOLO26n` · `Ultralytics` · `OpenCV` · `Streamlit` · `PyTorch` · `Arduino` · `Computer Vision` · `Robotics`

---

## 🎯 V1 Goal

Build and verify a complete prototype capable of detecting and sorting four recyclable waste categories:

* Plastic
* Metal
* Glass
* Paper / Cardboard

Objects that do not satisfy the required detection confidence will later be handled by the Python decision layer and routed to the **Reject** path.

---

## V1 Waste Classes

The V1 YOLO model is frozen to four classes:

```text
0: plastic
1: metal
2: glass
3: paper_cardboard
```

`Reject` is not a YOLO class.

Unknown or low-confidence objects will be handled later by the Python decision logic and routed to the reject path.

---

## V1 Prototype Flow

The planned complete system flow is:

```text
Waste object
    ↓
Fixed camera
    ↓
YOLO26n detection
    ↓
Python decision logic
    ↓
Class / Reject decision
    ↓
PySerial
    ↓
Arduino-compatible controller
    ↓
Conveyor + sensor + robotic arm
    ↓
Target recycling bin
```

---

## Current Project Status

### Dataset

Completed:

- dataset preparation
- label validation
- class validation
- bounding-box validation
- duplicate-label analysis
- train/validation leakage analysis
- final train and validation split verification

The V1 dataset contains:

```text
Training images:   15,148
Training objects:  83,850
Validation images: 1,268
Validation objects: 2,086
```

Detailed dataset documentation:

```text
docs/DATASET_PREPARATION_V1.md
docs/DATASET_SOURCES.md
docs/LABELING_GUIDE.md
```

---

### YOLO26n Training

The first official YOLO26n V1 baseline has been trained successfully.

Training configuration:

```text
Model:       yolo26n.pt
Epochs:      80
Image size:  640
Batch size:  16
Device:      CUDA GPU
```

Final overall validation:

| Metric | Result |
|---|---:|
| Precision | 0.809 |
| Recall | 0.747 |
| mAP50 | 0.802 |
| mAP50-95 | 0.452 |

Detailed training documentation:

```text
docs/YOLO26N_TRAINING_V1.md
```

The trained model weights are stored locally and are intentionally excluded from Git.

---

## AI Inference

The AI inference layer is implemented in:

```text
ai/inference.py
```

It currently supports:

- loading the trained `best.pt` model
- validating the four V1 classes
- running YOLO inference on images
- extracting detected class names
- extracting confidence scores
- returning YOLO detection results for the user interface

---

## Streamlit V1 Image Tester

A Streamlit-based testing interface has been implemented in:

```text
ai/streamlit_app.py
```

The interface currently supports:

- uploading multiple waste images
- selecting one uploaded image for analysis
- displaying the original image
- displaying the YOLO annotated image
- displaying detected classes
- displaying confidence scores
- displaying detection results in a table
- counting currently uploaded images
- counting unique images tested during the session
- counting total detections during the session
- preventing the same image from being counted repeatedly after Streamlit reruns
- resetting the current testing session
- side-by-side original and YOLO result views

Detailed documentation:

```text
docs/AI_INFERENCE_STREAMLIT_V1.md
```

---

## Run the Streamlit Tester

Activate the project virtual environment first.

Then run:

```cmd
python -m streamlit run ai\\streamlit_app.py
```

Streamlit will start a local web interface, normally at:

```text
http://localhost:8501
```

Upload one or more real waste images and select an image to run inference.

---

## Run Automated Tests

Run the complete automated test suite with:

```cmd
python -m pytest -q
```

The test suite covers the current AI inference and Streamlit workflow, including:

- V1 class configuration
- trained model path
- model loading
- image inference
- detection extraction
- image preparation
- multi-image upload
- image selection
- session statistics
- duplicate session counting protection
- reset behavior
- interface layout

---

## Repository Structure

```text
Smart_Recycle_AI-Hub/
│
├── ai/
│   ├── inference.py
│   ├── streamlit_app.py
│   └── README.md
│
├── config/
├── data/
├── docs/
├── logs/
├── scripts/
├── tests/
│
├── requirements.txt
└── README.md
```

Large generated files and datasets are intentionally excluded from Git.

Examples include:

```text
*.pt
runs/
data/raw/
data/processed/
data/interim/
data/inspection/
.venv/
.env
```

---

## Important Model Location

The current trained V1 model is expected locally at:

```text
runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt
```

`best.pt` is not stored in the Git repository.

A collaborator who wants to run inference must therefore obtain the trained weights separately and place them in the expected local path, or update the model path configuration accordingly.

---

## Current AI Milestone

Completed:

```text
Dataset preparation        DONE
Dataset validation         DONE
YOLO26n training           DONE
Final model validation     DONE
best.pt generation         DONE
Python inference layer     DONE
Streamlit image tester     DONE
Multi-image testing UI     DONE
Automated tests            DONE
```

---

## Next Development Steps

The immediate development path is:

```text
Confidence threshold
        ↓
Best-detection / decision logic
        ↓
Reject handling
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

Real-world observations will be documented before deciding whether model retraining is necessary.

The project should not add new waste classes before the V1 prototype is integrated and tested successfully.

---

## Documentation

Main project documents include:

```text
docs/PROJECT_SCOPE_V1.md
docs/DATASET_PREPARATION_V1.md
docs/DATASET_SOURCES.md
docs/LABELING_GUIDE.md
docs/YOLO26N_TRAINING_V1.md
docs/AI_INFERENCE_STREAMLIT_V1.md
ai/README.md
```

---

## Project Principle

Build and verify the complete V1 pipeline before expanding the scope.

**We can do it. **
