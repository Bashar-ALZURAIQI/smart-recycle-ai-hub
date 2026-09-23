from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from PIL import Image


def make_record(
    builder,
    source: str,
    image_path: Path,
) -> object:
    label_path = image_path.with_suffix(".txt")

    label_path.write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    return builder.CandidateRecord(
        source=source,
        source_split="train",
        image_path=image_path,
        label_path=label_path,
        labels=[
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        is_negative=False,
    )


def test_exact_duplicate_keeps_higher_priority_source(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    v2_image = tmp_path / "v2.png"
    whitemind_image = tmp_path / "whitemind.png"

    image = Image.new(
        "RGB",
        (64, 48),
        color=(30, 90, 180),
    )

    image.save(v2_image)

    shutil.copyfile(
        v2_image,
        whitemind_image,
    )

    v2_record = make_record(
        builder,
        "waste_v2",
        v2_image,
    )

    whitemind_record = make_record(
        builder,
        "whitemind_yolo_waste",
        whitemind_image,
    )

    result = builder.deduplicate_records(
        [
            whitemind_record,
            v2_record,
        ],
        source_priority=[
            "waste_v2",
            "general_waste_data",
            "whitemind_yolo_waste",
            "waste_detection_dataset_1",
        ],
    )

    assert len(result.records) == 1

    assert (
        result.records[0].source
        == "waste_v2"
    )

    assert result.exact_removed == 1
    assert result.perceptual_removed == 0


def test_perceptual_duplicate_keeps_higher_priority_source(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    general_image = tmp_path / "general.png"
    dataset1_image = tmp_path / "dataset1.bmp"

    image = Image.new(
        "RGB",
        (72, 54),
    )

    for y in range(54):
        for x in range(72):
            image.putpixel(
                (x, y),
                (
                    (x * 3) % 256,
                    (y * 5) % 256,
                    ((x + y) * 7) % 256,
                ),
            )

    image.save(
        general_image,
        format="PNG",
    )

    image.save(
        dataset1_image,
        format="BMP",
    )

    general_fingerprint = builder.image_fingerprint(
        general_image,
    )

    dataset1_fingerprint = builder.image_fingerprint(
        dataset1_image,
    )

    assert (
        general_fingerprint.sha256
        != dataset1_fingerprint.sha256
    )

    assert (
        general_fingerprint.dhash
        == dataset1_fingerprint.dhash
    )

    assert (
        general_fingerprint.width
        == dataset1_fingerprint.width
    )

    assert (
        general_fingerprint.height
        == dataset1_fingerprint.height
    )

    general_record = make_record(
        builder,
        "general_waste_data",
        general_image,
    )

    dataset1_record = make_record(
        builder,
        "waste_detection_dataset_1",
        dataset1_image,
    )

    result = builder.deduplicate_records(
        [
            dataset1_record,
            general_record,
        ],
        source_priority=[
            "waste_v2",
            "general_waste_data",
            "whitemind_yolo_waste",
            "waste_detection_dataset_1",
        ],
    )

    assert len(result.records) == 1

    assert (
        result.records[0].source
        == "general_waste_data"
    )

    assert result.exact_removed == 0
    assert result.perceptual_removed == 1