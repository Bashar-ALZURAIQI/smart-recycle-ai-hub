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


def test_exact_benchmark_overlap_is_removed(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    benchmark_image = tmp_path / "benchmark.png"
    leaked_image = tmp_path / "leaked.png"
    safe_image = tmp_path / "safe.png"

    Image.new(
        "RGB",
        (64, 48),
        color=(40, 100, 180),
    ).save(benchmark_image)

    shutil.copyfile(
        benchmark_image,
        leaked_image,
    )

    Image.new(
        "RGB",
        (80, 60),
        color=(200, 50, 20),
    ).save(safe_image)

    leaked_record = make_record(
        builder,
        "whitemind_yolo_waste",
        leaked_image,
    )

    safe_record = make_record(
        builder,
        "general_waste_data",
        safe_image,
    )

    benchmark_fingerprints = [
        builder.image_fingerprint(
            benchmark_image
        )
    ]

    result = builder.protect_records_from_benchmarks(
        [
            leaked_record,
            safe_record,
        ],
        benchmark_fingerprints,
    )

    assert len(result.records) == 1

    assert (
        result.records[0].image_path
        == safe_image
    )

    assert result.exact_removed == 1
    assert result.perceptual_removed == 0


def test_perceptual_benchmark_overlap_is_removed(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    benchmark_image = tmp_path / "benchmark.png"
    leaked_image = tmp_path / "leaked.bmp"
    safe_image = tmp_path / "safe.png"

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
        benchmark_image,
        format="PNG",
    )

    image.save(
        leaked_image,
        format="BMP",
    )

    Image.new(
        "RGB",
        (90, 60),
        color=(15, 220, 70),
    ).save(safe_image)

    benchmark_fingerprint = builder.image_fingerprint(
        benchmark_image
    )

    leaked_fingerprint = builder.image_fingerprint(
        leaked_image
    )

    assert (
        benchmark_fingerprint.sha256
        != leaked_fingerprint.sha256
    )

    assert (
        benchmark_fingerprint.dhash
        == leaked_fingerprint.dhash
    )

    assert (
        benchmark_fingerprint.width
        == leaked_fingerprint.width
    )

    assert (
        benchmark_fingerprint.height
        == leaked_fingerprint.height
    )

    leaked_record = make_record(
        builder,
        "waste_detection_dataset_1",
        leaked_image,
    )

    safe_record = make_record(
        builder,
        "general_waste_data",
        safe_image,
    )

    result = builder.protect_records_from_benchmarks(
        [
            leaked_record,
            safe_record,
        ],
        [
            benchmark_fingerprint,
        ],
    )

    assert len(result.records) == 1

    assert (
        result.records[0].image_path
        == safe_image
    )

    assert result.exact_removed == 0
    assert result.perceptual_removed == 1