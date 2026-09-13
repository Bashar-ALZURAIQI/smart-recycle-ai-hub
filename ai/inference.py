from pathlib import Path

from ultralytics import YOLO


SUPPORTED_CLASSES = [
    "plastic",
    "metal",
    "glass",
    "paper_cardboard",
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "waste_v1"
    / "yolo26n_v1_baseline"
    / "weights"
    / "best.pt"
)


def load_model():
    return YOLO(str(MODEL_PATH))


def run_inference(model, image):
    return model.predict(
        source=image,
        verbose=False,
    )


def extract_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        detections.append(
            {
                "class": result.names[class_id],
                "confidence": confidence,
            }
        )

    return detections