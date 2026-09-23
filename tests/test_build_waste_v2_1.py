from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def make_whitemind_policy(builder, root=None):
    return builder.SourcePolicy(
        source_id="whitemind_yolo_waste",
        root=root,
        source_type="roboflow_yolo",
        class_map={
            2: 2,
            3: 1,
            5: 3,
            6: 0,
        },
        unsupported_class_ids={
            0,
            1,
            4,
        },
        preserve_empty_labels=False,
        reject_non_detection_labels=True,
        exclude_if_any_unsupported=True,
    )


def make_dataset1_policy(builder, root=None):
    return builder.SourcePolicy(
        source_id="waste_detection_dataset_1",
        root=root,
        source_type="roboflow_yolo",
        class_map={
            0: 1,
            1: 2,
            4: 3,
            9: 2,
            10: 2,
            11: 0,
            14: 3,
            15: 1,
            16: 1,
            18: 3,
            19: 3,
            20: 0,
            21: 0,
            22: 0,
            24: 3,
            25: 0,
            26: 0,
            29: 0,
        },
        unsupported_class_ids={
            2,
            3,
            5,
            6,
            7,
            8,
            12,
            13,
            17,
            23,
            27,
            28,
        },
        preserve_empty_labels=False,
        reject_non_detection_labels=True,
        exclude_if_any_unsupported=True,
    )


def make_v2_policy(builder, root=None):
    return builder.SourcePolicy(
        source_id="waste_v2",
        root=root,
        source_type="processed_yolo",
        class_map={
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        },
        unsupported_class_ids=set(),
        preserve_empty_labels=True,
        reject_non_detection_labels=True,
        exclude_if_any_unsupported=True,
    )


def test_whitemind_supported_labels_are_remapped() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    policy = make_whitemind_policy(builder)

    labels, reason = builder.remap_image_labels(
        [
            "6 0.5 0.5 0.2 0.2",
            "3 0.4 0.4 0.1 0.1",
            "2 0.3 0.3 0.2 0.2",
            "5 0.6 0.6 0.1 0.1",
        ],
        policy,
    )

    assert reason is None

    assert labels == [
        "0 0.50000000 0.50000000 0.20000000 0.20000000",
        "1 0.40000000 0.40000000 0.10000000 0.10000000",
        "2 0.30000000 0.30000000 0.20000000 0.20000000",
        "3 0.60000000 0.60000000 0.10000000 0.10000000",
    ]


def test_whitemind_image_with_unsupported_class_is_excluded() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    policy = make_whitemind_policy(builder)

    labels, reason = builder.remap_image_labels(
        [
            "6 0.5 0.5 0.2 0.2",
            "4 0.4 0.4 0.3 0.3",
        ],
        policy,
    )

    assert labels == []
    assert reason == "unsupported_class"


def test_dataset1_non_five_token_label_excludes_whole_image() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    policy = make_dataset1_policy(builder)

    labels, reason = builder.remap_image_labels(
        [
            "20 0.5 0.5 0.2 0.2",
            "26 0.10 0.10 0.20 0.10 0.30 0.20 0.20 0.30",
        ],
        policy,
    )

    assert labels == []
    assert reason == "non_detection_label"


def test_external_empty_label_is_excluded() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    policy = make_dataset1_policy(builder)

    labels, reason = builder.remap_image_labels(
        [],
        policy,
    )

    assert labels == []
    assert reason == "empty_label"


def test_v2_empty_label_is_preserved_as_negative() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    policy = make_v2_policy(builder)

    labels, reason = builder.remap_image_labels(
        [""],
        policy,
    )

    assert labels == []
    assert reason is None


def test_source_discovery_pairs_image_and_label(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    root = tmp_path / "whitemind"

    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image_path = images_dir / "plastic_sample.jpg"
    label_path = labels_dir / "plastic_sample.txt"

    image_path.write_bytes(
        b"fake-image-for-discovery-test"
    )

    label_path.write_text(
        "6 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    policy = make_whitemind_policy(
        builder,
        root=root,
    )

    records = builder.discover_source_records(
        policy,
    )

    assert len(records) == 1

    record = records[0]

    assert record.source == "whitemind_yolo_waste"
    assert record.source_split == "train"
    assert record.image_path == image_path
    assert record.label_path == label_path
    assert record.is_negative is False

    assert record.labels == [
        "0 0.50000000 0.50000000 0.20000000 0.20000000"
    ]


def test_family_id_normalizes_roboflow_and_augmentation_names() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    roboflow_family = builder.family_id_for_image(
        "whitemind_yolo_waste",
        Path("plastic_bottle.rf.abc123.jpg"),
    )

    augmented_family = builder.family_id_for_image(
        "waste_v2",
        Path("metal_can_aug_12.jpg"),
    )

    augmented_word_family = builder.family_id_for_image(
        "waste_v2",
        Path("glass_jar_augmented-7.png"),
    )

    assert (
        roboflow_family
        == "whitemind_yolo_waste:plastic_bottle"
    )

    assert (
        augmented_family
        == "waste_v2:metal_can"
    )

    assert (
        augmented_word_family
        == "waste_v2:glass_jar"
    )


def test_source_discovery_rejects_image_without_label(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    root = tmp_path / "whitemind"

    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    (
        images_dir / "orphan_image.jpg"
    ).write_bytes(
        b"fake-orphan-image"
    )

    policy = make_whitemind_policy(
        builder,
        root=root,
    )

    with pytest.raises(
        ValueError,
        match="Image/label mismatch",
    ):
        builder.discover_source_records(
            policy,
        )


def test_source_discovery_rejects_label_without_image(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    root = tmp_path / "whitemind"

    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    (
        labels_dir / "orphan_label.txt"
    ).write_text(
        "6 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    policy = make_whitemind_policy(
        builder,
        root=root,
    )

    with pytest.raises(
        ValueError,
        match="Image/label mismatch",
    ):
        builder.discover_source_records(
            policy,
        )