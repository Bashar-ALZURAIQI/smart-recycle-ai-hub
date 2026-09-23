from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from PIL import Image


def test_image_fingerprint_detects_exact_copy(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    original = tmp_path / "original.png"
    copied = tmp_path / "renamed_copy.png"

    image = Image.new(
        "RGB",
        (32, 24),
        color=(120, 80, 40),
    )

    image.save(original)

    shutil.copyfile(
        original,
        copied,
    )

    first = builder.image_fingerprint(
        original,
    )

    second = builder.image_fingerprint(
        copied,
    )

    assert first.sha256 == second.sha256
    assert first.dhash == second.dhash

    assert first.width == 32
    assert first.height == 24

    assert second.width == 32
    assert second.height == 24

    assert len(first.sha256) == 64
    assert isinstance(first.dhash, str)
    assert first.dhash


def test_benchmark_match_detects_exact_sha256_overlap(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    benchmark_image = tmp_path / "benchmark.png"
    candidate_image = tmp_path / "candidate.png"

    image = Image.new(
        "RGB",
        (40, 30),
        color=(10, 150, 220),
    )

    image.save(benchmark_image)

    shutil.copyfile(
        benchmark_image,
        candidate_image,
    )

    benchmark_fingerprint = builder.image_fingerprint(
        benchmark_image,
    )

    candidate_fingerprint = builder.image_fingerprint(
        candidate_image,
    )

    match_kind = builder.benchmark_match_kind(
        candidate_fingerprint,
        [benchmark_fingerprint],
    )

    assert match_kind == "exact"


def test_benchmark_match_detects_perceptual_exact_overlap(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    benchmark_image = tmp_path / "benchmark.png"
    candidate_image = tmp_path / "candidate.bmp"

    image = Image.new(
        "RGB",
        (48, 36),
        color=(90, 160, 210),
    )

    image.save(
        benchmark_image,
        format="PNG",
    )

    image.save(
        candidate_image,
        format="BMP",
    )

    benchmark_fingerprint = builder.image_fingerprint(
        benchmark_image,
    )

    candidate_fingerprint = builder.image_fingerprint(
        candidate_image,
    )

    assert (
        benchmark_fingerprint.sha256
        != candidate_fingerprint.sha256
    )

    assert (
        benchmark_fingerprint.dhash
        == candidate_fingerprint.dhash
    )

    assert (
        benchmark_fingerprint.width
        == candidate_fingerprint.width
    )

    assert (
        benchmark_fingerprint.height
        == candidate_fingerprint.height
    )

    match_kind = builder.benchmark_match_kind(
        candidate_fingerprint,
        [benchmark_fingerprint],
    )

    assert match_kind == "perceptual_exact"


def test_collect_benchmark_fingerprints_reads_images_recursively(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    benchmark_root = tmp_path / "benchmark"
    images_dir = benchmark_root / "nested" / "images"

    images_dir.mkdir(
        parents=True
    )

    first_image = images_dir / "first.png"
    second_image = images_dir / "second.jpg"

    Image.new(
        "RGB",
        (20, 10),
        color=(255, 0, 0),
    ).save(first_image)

    Image.new(
        "RGB",
        (30, 15),
        color=(0, 255, 0),
    ).save(second_image)

    (
        benchmark_root / "ignore.txt"
    ).write_text(
        "not an image",
        encoding="utf-8",
    )

    fingerprints = builder.collect_benchmark_fingerprints(
        [benchmark_root],
    )

    assert len(fingerprints) == 2

    dimensions = {
        (
            fingerprint.width,
            fingerprint.height,
        )
        for fingerprint in fingerprints
    }

    assert dimensions == {
        (20, 10),
        (30, 15),
    }


def test_collect_benchmark_fingerprints_allows_empty_root(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    empty_root = tmp_path / "empty_benchmark"

    empty_root.mkdir()

    fingerprints = builder.collect_benchmark_fingerprints(
        [empty_root],
    )

    assert fingerprints == []