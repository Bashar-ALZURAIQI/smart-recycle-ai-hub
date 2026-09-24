from __future__ import annotations

import importlib
from pathlib import Path


def test_load_build_config_reads_real_waste_v2_1_contract() -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    config_path = Path(
        "config/waste_v2_1_mapping.yaml"
    )

    config = builder.load_build_config(
        config_path
    )

    project_root = (
        config_path.resolve()
        .parent
        .parent
    )

    assert config.version == 1
    assert config.seed == 26
    assert config.valid_fraction == 0.12
    assert config.max_negative_fraction == 0.20

    assert (
        config.perceptual_dedupe_mode
        == "exact_dhash_dimensions"
    )

    assert config.source_priority == [
        "waste_v2",
        "general_waste_data",
        "whitemind_yolo_waste",
        "waste_detection_dataset_1",
    ]

    assert config.protected_benchmark_roots == [
        (
            project_root
            / "data"
            / "benchmarks"
            / "external_test_v1"
        ),
        (
            project_root
            / "data"
            / "benchmarks"
            / "reject_challenge_v1"
        ),
        (
            project_root
            / "data"
            / "benchmarks"
            / "real_world_v2_1"
        ),
    ]

    assert set(
        config.sources
    ) == {
        "waste_v2",
        "general_waste_data",
        "whitemind_yolo_waste",
        "waste_detection_dataset_1",
    }

    waste_v2 = config.sources[
        "waste_v2"
    ]

    assert waste_v2.source_id == "waste_v2"
    assert waste_v2.source_type == "processed_yolo"

    assert waste_v2.root == (
        project_root
        / "data"
        / "processed"
        / "waste_v2"
    )

    assert waste_v2.class_map == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }

    assert waste_v2.unsupported_class_ids == set()
    assert waste_v2.preserve_empty_labels is True
    assert waste_v2.reject_non_detection_labels is True
    assert waste_v2.exclude_if_any_unsupported is True

    whitemind = config.sources[
        "whitemind_yolo_waste"
    ]

    assert whitemind.class_map == {
        2: 2,
        3: 1,
        5: 3,
        6: 0,
    }

    assert whitemind.unsupported_class_ids == {
        0,
        1,
        4,
    }

    dataset_1 = config.sources[
        "waste_detection_dataset_1"
    ]

    assert dataset_1.class_map[0] == 1
    assert dataset_1.class_map[1] == 2
    assert dataset_1.class_map[11] == 0
    assert dataset_1.class_map[24] == 3
    assert dataset_1.class_map[29] == 0

    general = config.sources[
        "general_waste_data"
    ]

    assert general.class_map[1] == 3
    assert general.class_map[3] == 2
    assert general.class_map[11] == 1
    assert general.class_map[23] == 0
    assert general.class_map[31] == 0

    assert general.unsupported_class_ids == {
        0,
        2,
    }