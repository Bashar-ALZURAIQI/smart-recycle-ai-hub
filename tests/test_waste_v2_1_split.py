from __future__ import annotations

import importlib
from pathlib import Path


def make_record(
    builder,
    tmp_path: Path,
    name: str,
    is_negative: bool,
    source: str = "waste_v2",
):
    image_path = (
        tmp_path
        / f"{name}.jpg"
    )

    label_path = (
        tmp_path
        / f"{name}.txt"
    )

    labels = (
        []
        if is_negative
        else [
            "0 0.50000000 0.50000000 0.20000000 0.20000000"
        ]
    )

    return builder.CandidateRecord(
        source=source,
        source_split="train",
        image_path=image_path,
        label_path=label_path,
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

    result = builder.limit_negative_fraction(
        positives + negatives,
        max_negative_fraction=0.20,
        seed=26,
    )

    kept_negative_count = sum(
        1
        for record in result.records
        if record.is_negative
    )

    assert len(
        result.records
    ) == 10

    assert kept_negative_count == 2

    assert result.negatives_kept == 2
    assert result.negatives_removed == 6

    assert (
        kept_negative_count
        / len(result.records)
    ) <= 0.20


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
        for index in range(20)
    ]

    negatives = [
        make_record(
            builder,
            tmp_path,
            f"negative_{index}",
            True,
        )
        for index in range(20)
    ]

    records = (
        positives
        + negatives
    )

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

    different_seed = (
        builder.limit_negative_fraction(
            records,
            max_negative_fraction=0.20,
            seed=27,
        )
    )

    first_negative_names = {
        record.image_path.name
        for record in first.records
        if record.is_negative
    }

    second_negative_names = {
        record.image_path.name
        for record in second.records
        if record.is_negative
    }

    different_negative_names = {
        record.image_path.name
        for record in different_seed.records
        if record.is_negative
    }

    assert (
        first_negative_names
        == second_negative_names
    )

    assert (
        first_negative_names
        != different_negative_names
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
            "bottle.rf.first",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "bottle.rf.second",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "paper_aug1",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "paper_aug2",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "metal_one",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "glass_one",
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

    different_seed = (
        builder.family_safe_split(
            records,
            valid_fraction=0.50,
            seed=27,
        )
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

    assert (
        first_valid
        == second_valid
    )

    assert (
        first_valid
        != different_seed_valid
    )


def test_family_safe_split_represents_multiple_sources_when_practical(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    records = []

    for source in (
        "source_a",
        "source_b",
        "source_c",
    ):
        for index in range(3):
            records.append(
                make_record(
                    builder,
                    tmp_path,
                    f"{source}_family_{index}",
                    False,
                    source=source,
                )
            )

    result = builder.family_safe_split(
        records,
        valid_fraction=0.33,
        seed=26,
    )

    valid_sources = {
        record.source
        for record in result.valid_records
    }

    assert len(
        result.valid_records
    ) == 3

    assert valid_sources == {
        "source_a",
        "source_b",
        "source_c",
    }

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


def test_family_safe_split_keeps_single_family_in_train(
    tmp_path: Path,
) -> None:
    builder = importlib.import_module(
        "scripts.build_waste_v2_1"
    )

    records = [
        make_record(
            builder,
            tmp_path,
            "single.rf.first",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "single.rf.second",
            False,
        ),
        make_record(
            builder,
            tmp_path,
            "single.rf.third",
            False,
        ),
    ]

    result = builder.family_safe_split(
        records,
        valid_fraction=0.50,
        seed=26,
    )

    assert len(
        result.train_records
    ) == 3

    assert len(
        result.valid_records
    ) == 0

    train_families = {
        builder.family_id_for_image(
            record.source,
            record.image_path,
        )
        for record in result.train_records
    }

    assert train_families == {
        "waste_v2:single"
    }