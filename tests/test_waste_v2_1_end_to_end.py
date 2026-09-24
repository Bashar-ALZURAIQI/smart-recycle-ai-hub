from __future__ import annotations

import csv
import importlib
import json
import shutil
from pathlib import Path

from PIL import Image


def write_example(
    root: Path,
    split_name: str,
    name: str,
    size: tuple[int, int],
    labels: list[str],
    color: tuple[int, int, int],
) -> Path:
    images_dir = (
        root
        / split_name
        / "images"
    )

    labels_dir = (
        root
        / split_name
        / "labels"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        images_dir
        / name
    )

    label_path = (
        labels_dir
        / Path(name).with_suffix(
            ".txt"
        ).name
    )

    Image.new(
        "RGB",
        size,
        color=color,
    ).save(
        image_path
    )

    label_text = ""

    if labels:
        label_text = (
            "\n".join(
                labels
            )
            + "\n"
        )

    label_path.write_text(
        label_text,
        encoding="utf-8",
    )

    return image_path


def test_build_waste_v2_1_end_to_end_fixture(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    config_dir = (
        tmp_path
        / "config"
    )

    config_dir.mkdir()

    waste_v2_root = (
        tmp_path
        / "data"
        / "waste_v2"
    )

    external_root = (
        tmp_path
        / "data"
        / "external"
    )

    benchmark_root = (
        tmp_path
        / "data"
        / "benchmarks"
        / "protected"
    )

    benchmark_root.mkdir(
        parents=True
    )

    write_example(
        waste_v2_root,
        "train",
        "v2_plastic.jpg",
        (61, 41),
        [
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (20, 60, 100),
    )

    write_example(
        waste_v2_root,
        "train",
        "v2_metal.jpg",
        (62, 42),
        [
            "1 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (40, 80, 120),
    )

    write_example(
        waste_v2_root,
        "valid",
        "v2_glass.jpg",
        (63, 43),
        [
            "2 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (60, 100, 140),
    )

    write_example(
        waste_v2_root,
        "valid",
        "v2_paper.jpg",
        (64, 44),
        [
            "3 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (80, 120, 160),
    )

    write_example(
        waste_v2_root,
        "train",
        "v2_negative.jpg",
        (65, 45),
        [],
        (100, 140, 180),
    )

    write_example(
        external_root,
        "train",
        "family.rf.one.jpg",
        (71, 51),
        [
            "1 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (120, 40, 70),
    )

    write_example(
        external_root,
        "valid",
        "family.rf.two.jpg",
        (72, 52),
        [
            "2 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (140, 60, 90),
    )

    write_example(
        external_root,
        "test",
        "external_paper.jpg",
        (73, 53),
        [
            "4 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (160, 80, 110),
    )

    write_example(
        external_root,
        "train",
        "external_glass.jpg",
        (74, 54),
        [
            "3 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (180, 100, 130),
    )

    write_example(
        external_root,
        "train",
        "mixed.jpg",
        (75, 55),
        [
            "1 0.50000000 0.50000000 0.20000000 0.20000000",
            "9 0.30000000 0.30000000 0.10000000 0.10000000",
        ],
        (200, 120, 150),
    )

    protected_source = write_example(
        external_root,
        "test",
        "protected.jpg",
        (76, 56),
        [
            "3 0.50000000 0.50000000 0.20000000 0.20000000"
        ],
        (220, 140, 170),
    )

    shutil.copyfile(
        protected_source,
        benchmark_root
        / "protected_copy.jpg",
    )

    config_path = (
        config_dir
        / "waste_v2_1_mapping.yaml"
    )

    config_path.write_text(
        """
version: 1

seed: 26
valid_fraction: 0.33
max_negative_fraction: 0.20

perceptual_dedupe:
  mode: exact_dhash_dimensions

protected_benchmarks:
  - data/benchmarks/protected

source_priority:
  - waste_v2
  - external

sources:

  waste_v2:
    type: processed_yolo
    root: data/waste_v2
    preserve_empty_labels: true
    reject_non_detection_labels: true
    exclude_if_any_unsupported: true

    class_map:
      0: 0
      1: 1
      2: 2
      3: 3

    unsupported_class_ids: []

  external:
    type: roboflow_yolo
    root: data/external
    preserve_empty_labels: false
    reject_non_detection_labels: true
    exclude_if_any_unsupported: true

    class_map:
      1: 0
      2: 1
      3: 2
      4: 3

    unsupported_class_ids:
      - 9
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output_root = (
        tmp_path
        / "data"
        / "processed"
        / "waste_v2_1"
    )

    builder.build_waste_v2_1(
        config_path=config_path,
        output_root=output_root,
    )

    assert (
        output_root
        / "data.yaml"
    ).exists()

    assert (
        output_root
        / "manifest.csv"
    ).exists()

    assert (
        output_root
        / "build_report.json"
    ).exists()

    assert (
        output_root
        / "train"
        / "images"
    ).is_dir()

    assert (
        output_root
        / "train"
        / "labels"
    ).is_dir()

    assert (
        output_root
        / "valid"
        / "images"
    ).is_dir()

    assert (
        output_root
        / "valid"
        / "labels"
    ).is_dir()

    with (
        output_root
        / "manifest.csv"
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    assert rows

    source_images = {
        row[
            "source_image"
        ]
        for row in rows
    }

    assert (
        "mixed.jpg"
        not in source_images
    )

    assert (
        "protected.jpg"
        not in source_images
    )

    family_rows = [
        row
        for row in rows
        if row[
            "family_id"
        ] == "external:family"
    ]

    assert len(
        family_rows
    ) == 2

    assert len(
        {
            row[
                "final_split"
            ]
            for row in family_rows
        }
    ) == 1

    family_splits: dict[
        str,
        set[str],
    ] = {}

    for row in rows:
        family_splits.setdefault(
            row[
                "family_id"
            ],
            set(),
        ).add(
            row[
                "final_split"
            ]
        )

    assert all(
        len(
            splits
        ) == 1
        for splits in (
            family_splits.values()
        )
    )

    assert {
        row[
            "final_split"
        ]
        for row in rows
    } == {
        "train",
        "valid",
    }

    for row in rows:
        output_image = (
            output_root
            / row[
                "final_split"
            ]
            / "images"
            / row[
                "output_name"
            ]
        )

        output_label = (
            output_root
            / row[
                "final_split"
            ]
            / "labels"
            / Path(
                row[
                    "output_name"
                ]
            ).with_suffix(
                ".txt"
            ).name
        )

        assert (
            output_image.exists()
        )

        assert (
            output_label.exists()
        )

    report = json.loads(
        (
            output_root
            / "build_report.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        report[
            "counts"
        ][
            "final_records"
        ]
        == len(
            rows
        )
    )

    assert (
        report[
            "benchmark_removals"
        ][
            "exact"
        ]
        == 1
    )

    assert (
        report[
            "invariant_failures"
        ]
        == []
    )