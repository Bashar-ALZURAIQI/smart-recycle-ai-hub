from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ai.inference import (
    MODEL_PATH,
    SUPPORTED_CLASSES,
        extract_detections,
    load_model,
    run_inference,
)


def test_supported_classes_are_v1_classes():
    assert SUPPORTED_CLASSES == [
        "plastic",
        "metal",
        "glass",
        "paper_cardboard",
    ]


def test_model_path_points_to_existing_best_weights():
    assert isinstance(MODEL_PATH, Path)
    assert MODEL_PATH.name == "best.pt"
    assert MODEL_PATH.is_file()


def test_load_model_returns_trained_yolo_model():
    model = load_model()

    assert isinstance(model, YOLO)
    assert list(model.names.values()) == SUPPORTED_CLASSES


def test_run_inference_returns_one_result_for_one_image():
    model = load_model()

    image = np.zeros(
        (640, 640, 3),
        dtype=np.uint8,
    )

    results = run_inference(
        model=model,
        image=image,
    )

    assert len(results) == 1
    assert hasattr(results[0], "boxes")

def test_extract_detections_returns_structured_detections():
    model = load_model()

    image = np.zeros(
        (640, 640, 3),
        dtype=np.uint8,
    )

    results = run_inference(
        model=model,
        image=image,
    )

    detections = extract_detections(results[0])

    assert isinstance(detections, list)

    for detection in detections:
        assert set(detection.keys()) == {
            "class",
            "confidence",
        }

        assert detection["class"] in SUPPORTED_CLASSES
        assert 0.0 <= detection["confidence"] <= 1.0