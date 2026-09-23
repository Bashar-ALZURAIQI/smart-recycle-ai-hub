from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ai.inference import (
    MODEL_PATH,
    SUPPORTED_CLASSES,
    V1_MODEL_PATH,
    V2_MODEL_PATH,
    extract_detections,
    load_model,
    run_inference,
)


def test_supported_classes_are_expected_classes():
    assert SUPPORTED_CLASSES == [
        "plastic",
        "metal",
        "glass",
        "paper_cardboard",
    ]


def test_v1_model_path_points_to_existing_best_weights():
    assert isinstance(V1_MODEL_PATH, Path)
    assert V1_MODEL_PATH.name == "best.pt"
    assert V1_MODEL_PATH.is_file()


def test_v2_model_path_points_to_existing_best_weights():
    assert isinstance(V2_MODEL_PATH, Path)
    assert V2_MODEL_PATH.name == "best.pt"
    assert V2_MODEL_PATH.is_file()


def test_default_model_path_is_v2():
    assert MODEL_PATH == V2_MODEL_PATH


def test_load_default_model_returns_v2_yolo_model():
    model = load_model()

    assert isinstance(model, YOLO)
    assert list(model.names.values()) == SUPPORTED_CLASSES


def test_load_v1_model_returns_yolo_model():
    model = load_model(V1_MODEL_PATH)

    assert isinstance(model, YOLO)
    assert list(model.names.values()) == SUPPORTED_CLASSES


def test_load_v2_model_returns_yolo_model():
    model = load_model(V2_MODEL_PATH)

    assert isinstance(model, YOLO)
    assert list(model.names.values()) == SUPPORTED_CLASSES


def test_run_inference_returns_one_result_for_one_image():
    model = load_model(V2_MODEL_PATH)

    image = np.zeros(
        (640, 640, 3),
        dtype=np.uint8,
    )

    results = run_inference(
        model=model,
        image=image,
        confidence_threshold=0.25,
    )

    assert len(results) == 1
    assert hasattr(results[0], "boxes")


def test_extract_detections_returns_structured_detections():
    model = load_model(V2_MODEL_PATH)

    image = np.zeros(
        (640, 640, 3),
        dtype=np.uint8,
    )

    results = run_inference(
        model=model,
        image=image,
        confidence_threshold=0.25,
    )

    detections = extract_detections(
        results[0]
    )

    assert isinstance(detections, list)

    for detection in detections:
        assert set(detection.keys()) == {
            "class",
            "confidence",
            "box",
        }

        assert detection["class"] in SUPPORTED_CLASSES

        assert (
            0.0
            <= detection["confidence"]
            <= 1.0
        )

        assert isinstance(
            detection["box"],
            list,
        )

        assert len(
            detection["box"]
        ) == 4


def test_extracted_detections_are_sorted_by_confidence():
    model = load_model(V2_MODEL_PATH)

    image = np.zeros(
        (640, 640, 3),
        dtype=np.uint8,
    )

    results = run_inference(
        model=model,
        image=image,
        confidence_threshold=0.05,
    )

    detections = extract_detections(
        results[0]
    )

    confidences = [
        detection["confidence"]
        for detection in detections
    ]

    assert confidences == sorted(
        confidences,
        reverse=True,
    )