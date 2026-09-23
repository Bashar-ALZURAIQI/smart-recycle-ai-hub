from pathlib import Path

from ultralytics import YOLO


SUPPORTED_CLASSES = [
    "plastic",
    "metal",
    "glass",
    "paper_cardboard",
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


V1_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "waste_v1"
    / "yolo26n_v1_baseline"
    / "weights"
    / "best.pt"
)


V2_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "waste_v2_yolo26n"
    / "weights"
    / "best.pt"
)


# Default model kept for backward compatibility with existing code/tests.
MODEL_PATH = V2_MODEL_PATH


def validate_model_classes(model):
    model_classes = [
        model.names[index]
        for index in sorted(model.names)
    ]

    if model_classes != SUPPORTED_CLASSES:
        raise RuntimeError(
            "Unexpected model classes. "
            f"Expected {SUPPORTED_CLASSES}, "
            f"but found {model_classes}."
        )


def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = YOLO(str(model_path))

    validate_model_classes(model)

    return model


def run_inference(
    model,
    image,
    confidence_threshold=0.25,
):
    return model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False,
    )


def extract_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        xyxy = box.xyxy[0].tolist()

        detections.append(
            {
                "class": result.names[class_id],
                "confidence": confidence,
                "box": [
                    round(float(value), 1)
                    for value in xyxy
                ],
            }
        )

    detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    return detections