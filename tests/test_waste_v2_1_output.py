from __future__ import annotations

import csv
import importlib
import json
import shutil
from pathlib import Path

from PIL import Image


def make_record(
    builder,
    source: str,
    image_path: Path,
    labels: list[str] | None = None,
    source_split: str = "train",
) -> object:
    label_path = image_path.with_suffix(".txt")

    label_path.write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    if labels is None:
        labels = [
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ]

    return builder.CandidateRecord(
        source=source,
        source_split=source_split,
        image_path=image_path,
        label_path=label_path,
        labels=labels,
        is_negative=not labels,
    )


def test_output_filename_is_content_based_and_portable(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    first_dir = tmp_path / "machine_a"
    second_dir = tmp_path / "machine_b"

    first_dir.mkdir()
    second_dir.mkdir()

    first_image = first_dir / "original_name.png"
    second_image = second_dir / "different_name.png"

    Image.new(
        "RGB",
        (64, 48),
        color=(30, 120, 210),
    ).save(first_image)

    shutil.copyfile(
        first_image,
        second_image,
    )

    first_record = make_record(
        builder,
        "waste_v2",
        first_image,
    )

    second_record = make_record(
        builder,
        "waste_v2",
        second_image,
    )

    first_name = builder.output_filename_for_record(
        first_record
    )

    second_name = builder.output_filename_for_record(
        second_record
    )

    assert first_name == second_name

    assert first_name.startswith(
        "waste_v2__"
    )

    assert first_name.endswith(
        ".png"
    )

    assert "machine_a" not in first_name
    assert "machine_b" not in first_name
    assert "original_name" not in first_name
    assert "different_name" not in first_name


def test_write_split_records_copies_image_and_writes_labels(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    source_image = source_dir / "paper.png"

    Image.new(
        "RGB",
        (80, 60),
        color=(220, 210, 180),
    ).save(source_image)

    remapped_labels = [
        "3 0.50000000 0.50000000 0.40000000 0.30000000",
        "0 0.20000000 0.30000000 0.10000000 0.15000000",
    ]

    record = make_record(
        builder,
        "general_waste_data",
        source_image,
        labels=remapped_labels,
    )

    output_root = tmp_path / "waste_v2_1"

    builder.write_split_records(
        [record],
        output_root=output_root,
        split_name="train",
    )

    output_name = builder.output_filename_for_record(
        record
    )

    output_image = (
        output_root
        / "train"
        / "images"
        / output_name
    )

    output_label = (
        output_root
        / "train"
        / "labels"
        / Path(output_name).with_suffix(".txt").name
    )

    assert output_image.exists()
    assert output_label.exists()

    assert (
        output_image.read_bytes()
        == source_image.read_bytes()
    )

    assert output_label.read_text(
        encoding="utf-8"
    ) == (
        "3 0.50000000 0.50000000 0.40000000 0.30000000\n"
        "0 0.20000000 0.30000000 0.10000000 0.15000000\n"
    )


def test_write_data_yaml_uses_portable_paths_and_four_classes(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    output_root = tmp_path / "waste_v2_1"

    builder.write_data_yaml(
        output_root
    )

    data_yaml = output_root / "data.yaml"

    assert data_yaml.exists()

    content = data_yaml.read_text(
        encoding="utf-8"
    )

    assert "path: ." in content
    assert "train: train/images" in content
    assert "val: valid/images" in content

    assert "0: plastic" in content
    assert "1: metal" in content
    assert "2: glass" in content
    assert "3: paper_cardboard" in content

    assert "reject" not in content.lower()
    assert str(tmp_path) not in content


def test_write_manifest_records_final_dataset_identity(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    train_image = source_dir / "plastic.png"
    valid_image = source_dir / "negative.jpg"

    Image.new(
        "RGB",
        (64, 48),
        color=(25, 100, 200),
    ).save(train_image)

    Image.new(
        "RGB",
        (80, 60),
        color=(200, 80, 30),
    ).save(valid_image)

    train_record = make_record(
        builder,
        "waste_v2",
        train_image,
        labels=[
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        source_split="train",
    )

    valid_record = make_record(
        builder,
        "general_waste_data",
        valid_image,
        labels=[],
        source_split="test",
    )

    output_root = tmp_path / "waste_v2_1"

    builder.write_manifest(
        train_records=[
            train_record,
        ],
        valid_records=[
            valid_record,
        ],
        output_root=output_root,
    )

    manifest_path = (
        output_root
        / "manifest.csv"
    )

    assert manifest_path.exists()

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    assert len(rows) == 2

    assert set(
        rows[0].keys()
    ) == {
        "source",
        "source_split",
        "family_id",
        "is_negative",
        "source_image",
        "output_name",
        "final_split",
        "sha256",
        "dhash",
        "width",
        "height",
    }

    train_row = next(
        row
        for row in rows
        if row["final_split"] == "train"
    )

    valid_row = next(
        row
        for row in rows
        if row["final_split"] == "valid"
    )

    assert train_row["source"] == "waste_v2"
    assert train_row["source_split"] == "train"
    assert train_row["source_image"] == "plastic.png"
    assert train_row["is_negative"] == "false"

    assert valid_row["source"] == "general_waste_data"
    assert valid_row["source_split"] == "test"
    assert valid_row["source_image"] == "negative.jpg"
    assert valid_row["is_negative"] == "true"

    assert train_row["output_name"].startswith(
        "waste_v2__"
    )

    assert valid_row["output_name"].startswith(
        "general_waste_data__"
    )

    assert len(
        train_row["sha256"]
    ) == 64

    assert len(
        valid_row["sha256"]
    ) == 64

    assert train_row["family_id"] == (
        "waste_v2:plastic"
    )

    assert valid_row["family_id"] == (
        "general_waste_data:negative"
    )

    content = manifest_path.read_text(
        encoding="utf-8"
    )

    assert str(tmp_path) not in content


def test_write_build_report_records_pipeline_counts_deterministically(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    train_image = source_dir / "plastic.png"
    valid_image = source_dir / "negative.jpg"

    Image.new(
        "RGB",
        (64, 48),
        color=(20, 110, 210),
    ).save(train_image)

    Image.new(
        "RGB",
        (80, 60),
        color=(180, 70, 40),
    ).save(valid_image)

    train_record = make_record(
        builder,
        "waste_v2",
        train_image,
        labels=[
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
    )

    valid_record = make_record(
        builder,
        "general_waste_data",
        valid_image,
        labels=[],
        source_split="test",
    )

    output_root = tmp_path / "waste_v2_1"

    builder.write_build_report(
        output_root=output_root,
        candidate_count=20,
        benchmark_exact_removed=1,
        benchmark_perceptual_removed=2,
        dedup_exact_removed=3,
        dedup_perceptual_removed=4,
        negatives_kept=1,
        negatives_removed=5,
        train_records=[
            train_record,
        ],
        valid_records=[
            valid_record,
        ],
    )

    report_path = (
        output_root
        / "build_report.json"
    )

    assert report_path.exists()

    first_content = report_path.read_text(
        encoding="utf-8"
    )

    report = json.loads(
        first_content
    )

    assert report["version"] == "2.1"

    assert report["classes"] == {
        "0": "plastic",
        "1": "metal",
        "2": "glass",
        "3": "paper_cardboard",
    }

    assert report["counts"] == {
        "candidate_records": 20,
        "final_records": 2,
        "train_records": 1,
        "valid_records": 1,
    }

    assert report["benchmark_removals"] == {
        "exact": 1,
        "perceptual": 2,
    }

    assert report["dedup_removals"] == {
        "exact": 3,
        "perceptual": 4,
    }

    assert report["negatives"] == {
        "kept": 1,
        "removed": 5,
    }

    assert report["final_sources"] == {
        "general_waste_data": 1,
        "waste_v2": 1,
    }

    assert report["final_class_annotations"] == {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 0,
    }

    assert "generated_at" not in report
    assert str(tmp_path) not in first_content

    builder.write_build_report(
        output_root=output_root,
        candidate_count=20,
        benchmark_exact_removed=1,
        benchmark_perceptual_removed=2,
        dedup_exact_removed=3,
        dedup_perceptual_removed=4,
        negatives_kept=1,
        negatives_removed=5,
        train_records=[
            train_record,
        ],
        valid_records=[
            valid_record,
        ],
    )

    second_content = report_path.read_text(
        encoding="utf-8"
    )

    assert first_content == second_content