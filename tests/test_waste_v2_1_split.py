from __future__ import annotations

import importlib
from pathlib import Path


def make_record(
    builder,
    tmp_path: Path,
    name: str,
    is_negative: bool,
    source: str = "waste_v2",
) -> object:
    labels = []

    if not is_negative:
        labels = [
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ]

    return builder.CandidateRecord(
        source=source,
        source_split="train",
        image_path=tmp_path / f"{name}.jpg",
        label_path=tmp_path / f"{name}.txt",
        labels=labels,
        is_negative=is_negative,
    )


def test_negative_fraction_is_capped_at_twenty_percent(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    positives = [
        make_record(
            builder,
            tmp_path,
            f"positive_{index}",
            False,
        )
        for index in range(4)
    ]

    negatives = [
        make_record(
            builder,
            tmp_path,
            f"negative_{index}",
            True,
        )
        for index in range(4)
    ]

    result = builder.limit_negative_fraction(
        positives + negatives,
        max_negative_fraction=0.20,
        seed=26,
    )

    kept_positives = [
        record
        for record in result.records
        if not record.is_negative
    ]

    kept_negatives = [
        record
        for record in result.records
        if record.is_negative
    ]

    assert len(result.records) == 5

    assert len(kept_positives) == 4
    assert len(kept_negatives) == 1

    assert result.negatives_kept == 1
    assert result.negatives_removed == 3

    assert (
        len(kept_negatives)
        / len(result.records)
        <= 0.20
    )


def test_negative_selection_is_deterministic_by_seed(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    positives = [
        make_record(
            builder,
            tmp_path,
            f"positive_{index}",
            False,
        )
        for index in range(8)
    ]

    negatives = [
        make_record(
            builder,
            tmp_path,
            f"negative_{index}",
            True,
        )
        for index in range(8)
    ]

    records = positives + negatives

    first = builder.limit_negative_fraction(
        records,
        max_negative_fraction=0.20,
        seed=26,
    )

    second = builder.limit_negative_fraction(
        records,
        max_negative_fraction=0.20,
        seed=26,
    )

    different_seed = builder.limit_negative_fraction(
        records,
        max_negative_fraction=0.20,
        seed=27,
    )

    first_negatives = {
        record.image_path.name
        for record in first.records
        if record.is_negative
    }

    second_negatives = {
        record.image_path.name
        for record in second.records
        if record.is_negative
    }

    different_seed_negatives = {
        record.image_path.name
        for record in different_seed.records
        if record.is_negative
    }

    assert len(first_negatives) == 2
    assert len(second_negatives) == 2
    assert len(different_seed_negatives) == 2

    assert first_negatives == second_negatives

    assert (
        first_negatives
        != different_seed_negatives
    )


def test_family_safe_split_never_separates_related_images(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    records = [
        make_record(
            builder,
            tmp_path,
            "bottle",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "bottle_aug_1",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "bottle.rf.abcdef",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "can",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "can_augmented_2",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "paper",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "glass",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "metal",
            False,
        ),
    ]

    result = builder.family_safe_split(
        records,
        valid_fraction=0.50,
        seed=26,
    )

    train_families = {
        builder.family_id_for_image(
            record.source,
            record.image_path,
        )
        for record in result.train_records
    }

    valid_families = {
        builder.family_id_for_image(
            record.source,
            record.image_path,
        )
        for record in result.valid_records
    }

    assert train_families.isdisjoint(
        valid_families
    )

    bottle_locations = {
        "train"
        if record in result.train_records
        else "valid"
        for record in records[:3]
    }

    can_locations = {
        "train"
        if record in result.train_records
        else "valid"
        for record in records[3:5]
    }

    assert len(bottle_locations) == 1
    assert len(can_locations) == 1

    assert (
        len(result.train_records)
        + len(result.valid_records)
        == len(records)
    )


def test_family_safe_split_is_deterministic_by_seed(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    records = [
        make_record(
            builder,
            tmp_path,
            f"family_{index}",
            False,
        )
        for index in range(12)
    ]

    first = builder.family_safe_split(
        records,
        valid_fraction=0.50,
        seed=26,
    )

    second = builder.family_safe_split(
        records,
        valid_fraction=0.50,
        seed=26,
    )

    different_seed = builder.family_safe_split(
        records,
        valid_fraction=0.50,
        seed=27,
    )

    first_valid = {
        record.image_path.name
        for record in first.valid_records
    }

    second_valid = {
        record.image_path.name
        for record in second.valid_records
    }

    different_seed_valid = {
        record.image_path.name
        for record in different_seed.valid_records
    }

    assert len(first_valid) == 6
    assert len(second_valid) == 6
    assert len(different_seed_valid) == 6

    assert first_valid == second_valid

    assert (
        first_valid
        != different_seed_valid
    )