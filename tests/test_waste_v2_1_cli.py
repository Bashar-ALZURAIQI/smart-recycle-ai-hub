from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from PIL import Image


def write_example(
    root: Path,
    split_name: str,
    name: str,
    size: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
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

    label_path.write_text(
        label + "\n",
        encoding="utf-8",
    )


def test_cli_builds_dataset_using_default_paths(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    script_path = Path(
        builder.__file__
    ).resolve()

    source_root = (
        tmp_path
        / "data"
        / "source"
    )

    write_example(
        source_root,
        "train",
        "plastic.jpg",
        (61, 41),
        "0 0.50000000 0.50000000 0.20000000 0.20000000",
        (30, 90, 150),
    )

    write_example(
        source_root,
        "valid",
        "metal.jpg",
        (63, 43),
        "1 0.50000000 0.50000000 0.20000000 0.20000000",
        (150, 90, 30),
    )

    config_dir = (
        tmp_path
        / "config"
    )

    config_dir.mkdir()

    config_path = (
        config_dir
        / "waste_v2_1_mapping.yaml"
    )

    config_path.write_text(
        """
version: 1

seed: 26
valid_fraction: 0.50
max_negative_fraction: 0.20

perceptual_dedupe:
  mode: exact_dhash_dimensions

protected_benchmarks: []

source_priority:
  - source

sources:

  source:
    type: processed_yolo
    root: data/source
    preserve_empty_labels: true
    reject_non_detection_labels: true
    exclude_if_any_unsupported: true

    class_map:
      0: 0
      1: 1
      2: 2
      3: 3

    unsupported_class_ids: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                script_path
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "STDOUT:\n"
        + result.stdout
        + "\nSTDERR:\n"
        + result.stderr
    )

    output_root = (
        tmp_path
        / "data"
        / "processed"
        / "waste_v2_1"
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
        / "valid"
        / "images"
    ).is_dir()