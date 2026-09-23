from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "waste_v2"
)

FINAL_CLASS_NAMES = {
    0: "plastic",
    1: "metal",
    2: "glass",
    3: "paper_cardboard",
}


SOURCE_CONFIGS = {
    "waste_v1": {
        "root": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "waste_v1"
        ),
        "class_map": {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        },
    },

    "garbage_classification": {
        "root": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "external_dataset_2"
            / "GARBAGE CLASSIFICATION"
        ),
        "class_map": {
            0: None,  # BIODEGRADABLE
            1: 3,     # CARDBOARD
            2: 2,     # GLASS
            3: 1,     # METAL
            4: 3,     # PAPER
            5: 0,     # PLASTIC
        },
    },

    "recyclable_waste_detection": {
        "root": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "recyclable_waste_detection"
        ),
        "class_map": {
            0: 3,  # Cardboard
            1: 2,  # Glass
            2: 1,  # Metal
            3: 3,  # Paper
            4: 0,  # Plastic
        },
    },

    "conveyor_waste_belt": {
        "root": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "conveyor_waste_belt"
        ),
        "class_map": {
            0: 2,     # Glass
            1: 1,     # Metal-Other
            2: None,  # Organic Food
            3: 3,     # Paper-Cardboard
            4: 0,     # Plastic
        },
    },
}


PROTECTED_BENCHMARKS = [
    (
        PROJECT_ROOT
        / "data"
        / "benchmarks"
        / "external_test_v1"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "benchmarks"
        / "reject_challenge_v1"
    ),
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def split_records_by_family(
    records: list[dict],
    valid_fraction: float = 0.12,
    seed: int = 26,
) -> tuple[list[dict], list[dict]]:

    if not 0 < valid_fraction < 1:
        raise ValueError(
            "valid_fraction must be between 0 and 1"
        )

    if not records:
        return [], []

    families = defaultdict(list)

    for record in records:
        families[
            record["family_id"]
        ].append(
            record
        )

    family_ids = list(
        families.keys()
    )

    if len(family_ids) < 2:
        return list(records), []

    target_valid_count = max(
        1,
        round(
            len(family_ids)
            * valid_fraction
        ),
    )

    target_valid_count = min(
        target_valid_count,
        len(family_ids) - 1,
    )

    source_to_families = defaultdict(list)
    family_to_source = {}

    for (
        family_id,
        family_records,
    ) in families.items():

        source = family_records[0][
            "source"
        ]

        family_to_source[
            family_id
        ] = source

        source_to_families[
            source
        ].append(
            family_id
        )

    rng = random.Random(
        seed
    )

    sources = sorted(
        source_to_families.keys()
    )

    rng.shuffle(
        sources
    )

    for source in sources:
        rng.shuffle(
            source_to_families[
                source
            ]
        )

    valid_family_ids = set()

    eligible_sources = [
        source
        for source in sources
        if len(
            source_to_families[
                source
            ]
        ) >= 2
    ]

    if (
        target_valid_count
        >= len(
            eligible_sources
        )
    ):
        for source in eligible_sources:

            valid_family_ids.add(
                source_to_families[
                    source
                ][0]
            )

    shuffled_families = list(
        family_ids
    )

    rng.shuffle(
        shuffled_families
    )

    for family_id in shuffled_families:

        if (
            len(valid_family_ids)
            >= target_valid_count
        ):
            break

        if (
            family_id
            in valid_family_ids
        ):
            continue

        source = family_to_source[
            family_id
        ]

        source_families = (
            source_to_families[
                source
            ]
        )

        selected_count = sum(
            1
            for selected
            in valid_family_ids
            if family_to_source[
                selected
            ]
            == source
        )

        maximum_valid = max(
            0,
            len(
                source_families
            )
            - 1,
        )

        if (
            selected_count
            >= maximum_valid
        ):
            continue

        valid_family_ids.add(
            family_id
        )

    if (
        len(valid_family_ids)
        < target_valid_count
    ):
        for family_id in shuffled_families:

            if (
                len(valid_family_ids)
                >= target_valid_count
            ):
                break

            if (
                family_id
                in valid_family_ids
            ):
                continue

            if (
                len(valid_family_ids)
                >= len(family_ids) - 1
            ):
                break

            valid_family_ids.add(
                family_id
            )

    train_records = []
    valid_records = []

    for (
        family_id,
        family_records,
    ) in families.items():

        if (
            family_id
            in valid_family_ids
        ):

            valid_records.extend(
                family_records
            )

        else:

            train_records.extend(
                family_records
            )

    return (
        train_records,
        valid_records,
    )


def select_controlled_negatives(
    records: list[dict],
    max_negative_fraction: float = 0.20,
    seed: int = 26,
) -> list[dict]:

    if not (
        0
        <= max_negative_fraction
        < 1
    ):
        raise ValueError(
            "max_negative_fraction must be "
            "between 0 and 1"
        )

    positives = [
        record
        for record in records
        if not record.get(
            "is_negative",
            False,
        )
    ]

    negatives = [
        record
        for record in records
        if record.get(
            "is_negative",
            False,
        )
    ]

    if not positives:
        return []

    if (
        max_negative_fraction
        == 0
    ):
        return positives

    maximum_negatives = math.floor(
        (
            len(positives)
            * max_negative_fraction
        )
        / (
            1
            - max_negative_fraction
        )
    )

    maximum_negatives = min(
        maximum_negatives,
        len(negatives),
    )

    rng = random.Random(
        seed
    )

    negatives = list(
        negatives
    )

    rng.shuffle(
        negatives
    )

    return (
        positives
        + negatives[
            :maximum_negatives
        ]
    )


def compute_dhash(
    image: Image.Image,
) -> str:

    grayscale = image.convert(
        "L"
    )

    resized = grayscale.resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )

    if hasattr(
        resized,
        "get_flattened_data",
    ):

        pixels = list(
            resized.get_flattened_data()
        )

    else:

        pixels = list(
            resized.getdata()
        )

    value = 0

    for row in range(8):

        offset = row * 9

        for column in range(8):

            left = pixels[
                offset + column
            ]

            right = pixels[
                offset + column + 1
            ]

            bit = (
                1
                if left > right
                else 0
            )

            value = (
                value << 1
            ) | bit

    return f"{value:016x}"


def image_fingerprint(
    path: Path,
) -> dict:

    sha256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    with Image.open(
        path
    ) as image:

        width, height = (
            image.size
        )

        dhash = compute_dhash(
            image
        )

    return {
        "sha256": sha256,
        "dhash": dhash,
        "width": width,
        "height": height,
    }


def family_id_for_image(
    source: str,
    image_path: Path,
) -> str:

    stem = image_path.stem

    if ".rf." in stem:

        stem = stem.split(
            ".rf.",
            1,
        )[0]

    stem = re.sub(
        r"(?i)[_-]aug(?:mented)?[_-]?\d+$",
        "",
        stem,
    )

    return (
        f"{source}::{stem.lower()}"
    )


def parse_yolo_annotation(
    line: str,
) -> tuple[int, str] | None:

    parts = (
        line.strip().split()
    )

    if not parts:
        return None

    try:

        class_id = int(
            float(
                parts[0]
            )
        )

        values = [
            float(value)
            for value
            in parts[1:]
        ]

    except ValueError:

        return None

    if len(values) == 4:

        (
            x_center,
            y_center,
            width,
            height,
        ) = values

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            return None

        text = (
            f"{x_center:.8f} "
            f"{y_center:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

        return (
            class_id,
            text,
        )

    if (
        len(values) >= 6
        and len(values) % 2 == 0
    ):

        xs = values[
            0::2
        ]

        ys = values[
            1::2
        ]

        if not all(
            0 <= coordinate <= 1
            for coordinate
            in values
        ):
            return None

        x_min = min(
            xs
        )

        x_max = max(
            xs
        )

        y_min = min(
            ys
        )

        y_max = max(
            ys
        )

        width = (
            x_max
            - x_min
        )

        height = (
            y_max
            - y_min
        )

        if (
            width <= 0
            or height <= 0
        ):
            return None

        x_center = (
            x_min
            + x_max
        ) / 2

        y_center = (
            y_min
            + y_max
        ) / 2

        text = (
            f"{x_center:.8f} "
            f"{y_center:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

        return (
            class_id,
            text,
        )

    return None


def resolve_dataset_splits(
    dataset_root: Path,
) -> list[tuple[str, Path]]:

    dataset_root = Path(
        dataset_root
    ).resolve()

    data_yaml = (
        dataset_root
        / "data.yaml"
    )

    if not data_yaml.exists():

        raise FileNotFoundError(
            f"Missing data.yaml: "
            f"{data_yaml}"
        )

    configuration = yaml.safe_load(
        data_yaml.read_text(
            encoding="utf-8"
        )
    )

    base_directory = (
        dataset_root
    )

    yaml_base = (
        configuration.get(
            "path"
        )
    )

    if yaml_base:

        yaml_base_path = Path(
            str(
                yaml_base
            )
        )

        if (
            yaml_base_path.is_absolute()
        ):

            base_directory = (
                yaml_base_path.resolve()
            )

        else:

            base_directory = (
                dataset_root
                / yaml_base_path
            ).resolve()

    results = []

    for key in (
        "train",
        "val",
        "valid",
        "test",
    ):

        value = (
            configuration.get(
                key
            )
        )

        if not value:
            continue

        values = (
            value
            if isinstance(
                value,
                list,
            )
            else [value]
        )

        for item in values:

            raw_path = Path(
                str(
                    item
                )
            )

            candidate_paths = []

            if (
                raw_path.is_absolute()
            ):

                candidate_paths.append(
                    raw_path.resolve()
                )

            else:

                literal_path = (
                    base_directory
                    / raw_path
                ).resolve()

                candidate_paths.append(
                    literal_path
                )

                cleaned_parts = list(
                    raw_path.parts
                )

                while (
                    cleaned_parts
                    and cleaned_parts[0]
                    in (
                        ".",
                        "..",
                    )
                ):

                    cleaned_parts.pop(
                        0
                    )

                if cleaned_parts:

                    fallback_path = (
                        dataset_root
                        / Path(
                            *cleaned_parts
                        )
                    ).resolve()

                    if (
                        fallback_path
                        not in candidate_paths
                    ):

                        candidate_paths.append(
                            fallback_path
                        )

            resolved_path = None

            for candidate in (
                candidate_paths
            ):

                if candidate.is_dir():

                    resolved_path = (
                        candidate
                    )

                    break

            if (
                resolved_path
                is not None
            ):

                results.append(
                    (
                        key,
                        resolved_path,
                    )
                )

    return results


def label_path_for_image(
    image_path: Path,
    images_root: Path,
) -> Path:

    relative_path = (
        image_path.relative_to(
            images_root
        )
    )

    labels_root = (
        images_root.parent
        / "labels"
    )

    return (
        labels_root
        / relative_path.with_suffix(
            ".txt"
        )
    )


def discover_source_records(
    source: str,
    config: dict,
    counters: Counter,
) -> list[dict]:

    dataset_root = Path(
        config[
            "root"
        ]
    )

    class_map = (
        config[
            "class_map"
        ]
    )

    records = []

    split_paths = (
        resolve_dataset_splits(
            dataset_root
        )
    )

    for (
        source_split,
        images_root,
    ) in split_paths:

        image_paths = sorted(
            path
            for path
            in images_root.rglob(
                "*"
            )
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        for image_path in (
            image_paths
        ):

            counters[
                f"{source}:images_seen"
            ] += 1

            label_path = (
                label_path_for_image(
                    image_path,
                    images_root,
                )
            )

            if not (
                label_path.exists()
            ):

                counters[
                    f"{source}:missing_label"
                ] += 1

                continue

            lines = [
                line.strip()
                for line
                in label_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines()
                if line.strip()
            ]

            if not lines:

                counters[
                    f"{source}:empty_label_skipped"
                ] += 1

                continue

            mapped_annotations = []

            saw_ignored_class = False
            invalid = False

            for line in lines:

                parsed = (
                    parse_yolo_annotation(
                        line
                    )
                )

                if (
                    parsed is None
                ):

                    invalid = True
                    break

                (
                    source_class_id,
                    bbox_text,
                ) = parsed

                if (
                    source_class_id
                    not in class_map
                ):

                    invalid = True
                    break

                target_class_id = (
                    class_map[
                        source_class_id
                    ]
                )

                if (
                    target_class_id
                    is None
                ):

                    saw_ignored_class = True

                else:

                    mapped_annotations.append(
                        (
                            f"{target_class_id} "
                            f"{bbox_text}"
                        )
                    )

            if invalid:

                counters[
                    f"{source}:invalid_or_unknown"
                ] += 1

                continue

            if (
                mapped_annotations
                and saw_ignored_class
            ):

                counters[
                    f"{source}:ambiguous_mixed"
                ] += 1

                continue

            mapped_annotations = list(
                dict.fromkeys(
                    mapped_annotations
                )
            )

            is_negative = (
                not mapped_annotations
                and saw_ignored_class
            )

            if (
                not mapped_annotations
                and not is_negative
            ):
                continue

            label_text = ""

            if mapped_annotations:

                label_text = (
                    "\n".join(
                        mapped_annotations
                    )
                    + "\n"
                )

            try:

                relative_string = str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                )

            except ValueError:

                relative_string = str(
                    image_path
                )

            record_hash = (
                hashlib.sha1(
                    (
                        source
                        + "::"
                        + relative_string
                    ).encode(
                        "utf-8"
                    )
                ).hexdigest()[
                    :12
                ]
            )

            record_id = (
                f"{source}__"
                f"{record_hash}"
            )

            output_name = (
                f"{record_id}"
                f"{image_path.suffix.lower()}"
            )

            records.append(
                {
                    "id": (
                        record_id
                    ),
                    "source": (
                        source
                    ),
                    "source_split": (
                        source_split
                    ),
                    "family_id": (
                        family_id_for_image(
                            source,
                            image_path,
                        )
                    ),
                    "image_path": (
                        image_path
                    ),
                    "output_name": (
                        output_name
                    ),
                    "label_text": (
                        label_text
                    ),
                    "is_negative": (
                        is_negative
                    ),
                }
            )

    return records


def build_protected_fingerprints(
    counters: Counter,
) -> tuple[
    set[str],
    set[tuple],
]:

    sha256_values = set()
    perceptual_values = set()

    for directory in (
        PROTECTED_BENCHMARKS
    ):

        if not (
            directory.exists()
        ):
            continue

        for image_path in (
            directory.rglob(
                "*"
            )
        ):

            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            try:

                fingerprint = (
                    image_fingerprint(
                        image_path
                    )
                )

            except Exception:

                counters[
                    "benchmark:unreadable"
                ] += 1

                continue

            sha256_values.add(
                fingerprint[
                    "sha256"
                ]
            )

            perceptual_values.add(
                (
                    fingerprint[
                        "dhash"
                    ],
                    fingerprint[
                        "width"
                    ],
                    fingerprint[
                        "height"
                    ],
                )
            )

    return (
        sha256_values,
        perceptual_values,
    )


def deduplicate_and_exclude_benchmarks(
    records: list[dict],
    counters: Counter,
) -> list[dict]:

    (
        benchmark_sha,
        benchmark_perceptual,
    ) = (
        build_protected_fingerprints(
            counters
        )
    )

    seen_sha = set()
    seen_perceptual = set()

    accepted = []

    for record in records:

        try:

            fingerprint = (
                image_fingerprint(
                    record[
                        "image_path"
                    ]
                )
            )

        except Exception:

            counters[
                "candidate:unreadable"
            ] += 1

            continue

        perceptual_key = (
            fingerprint[
                "dhash"
            ],
            fingerprint[
                "width"
            ],
            fingerprint[
                "height"
            ],
        )

        if (
            fingerprint[
                "sha256"
            ]
            in benchmark_sha
        ):

            counters[
                "excluded:benchmark_exact"
            ] += 1

            continue

        if (
            perceptual_key
            in benchmark_perceptual
        ):

            counters[
                "excluded:benchmark_perceptual"
            ] += 1

            continue

        if (
            fingerprint[
                "sha256"
            ]
            in seen_sha
        ):

            counters[
                "excluded:duplicate_exact"
            ] += 1

            continue

        if (
            perceptual_key
            in seen_perceptual
        ):

            counters[
                "excluded:duplicate_perceptual"
            ] += 1

            continue

        seen_sha.add(
            fingerprint[
                "sha256"
            ]
        )

        seen_perceptual.add(
            perceptual_key
        )

        record.update(
            fingerprint
        )

        accepted.append(
            record
        )

    return accepted


def _write_split(
    records: list[dict],
    split_name: str,
    output_directory: Path,
) -> None:

    images_directory = (
        output_directory
        / split_name
        / "images"
    )

    labels_directory = (
        output_directory
        / split_name
        / "labels"
    )

    images_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for record in records:

        source_image_path = Path(
            record[
                "image_path"
            ]
        )

        output_name = record.get(
            "output_name",
            source_image_path.name,
        )

        destination_image = (
            images_directory
            / output_name
        )

        destination_label = (
            labels_directory
            / (
                Path(
                    output_name
                ).stem
                + ".txt"
            )
        )

        shutil.copy2(
            source_image_path,
            destination_image,
        )

        destination_label.write_text(
            record.get(
                "label_text",
                "",
            ),
            encoding="utf-8",
        )


def _write_data_yaml(
    output_directory: Path,
) -> None:

    lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "",
        "names:",
    ]

    for (
        class_id,
        class_name,
    ) in (
        FINAL_CLASS_NAMES.items()
    ):

        lines.append(
            f"  {class_id}: "
            f"{class_name}"
        )

    lines.append(
        ""
    )

    (
        output_directory
        / "data.yaml"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def _write_manifest(
    train_records: list[dict],
    valid_records: list[dict],
    output_directory: Path,
) -> None:

    path = (
        output_directory
        / "manifest.csv"
    )

    fields = [
        "id",
        "split",
        "source",
        "source_split",
        "family_id",
        "is_negative",
        "source_image",
        "output_name",
        "sha256",
        "dhash",
        "width",
        "height",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for (
            split,
            records,
        ) in (
            (
                "train",
                train_records,
            ),
            (
                "valid",
                valid_records,
            ),
        ):

            for record in records:

                writer.writerow(
                    {
                        "id": record.get(
                            "id",
                            "",
                        ),
                        "split": (
                            split
                        ),
                        "source": record.get(
                            "source",
                            "",
                        ),
                        "source_split": record.get(
                            "source_split",
                            "",
                        ),
                        "family_id": record.get(
                            "family_id",
                            "",
                        ),
                        "is_negative": record.get(
                            "is_negative",
                            False,
                        ),
                        "source_image": str(
                            record.get(
                                "image_path",
                                "",
                            )
                        ),
                        "output_name": record.get(
                            "output_name",
                            "",
                        ),
                        "sha256": record.get(
                            "sha256",
                            "",
                        ),
                        "dhash": record.get(
                            "dhash",
                            "",
                        ),
                        "width": record.get(
                            "width",
                            "",
                        ),
                        "height": record.get(
                            "height",
                            "",
                        ),
                    }
                )


def _write_build_report(
    train_records: list[dict],
    valid_records: list[dict],
    output_directory: Path,
    counters: Counter | None = None,
) -> None:

    all_records = (
        list(
            train_records
        )
        + list(
            valid_records
        )
    )

    negative_count = sum(
        bool(
            record.get(
                "is_negative",
                False,
            )
        )
        for record
        in all_records
    )

    source_counts = Counter(
        record.get(
            "source",
            "unknown",
        )
        for record
        in all_records
    )

    train_source_counts = Counter(
        record.get(
            "source",
            "unknown",
        )
        for record
        in train_records
    )

    valid_source_counts = Counter(
        record.get(
            "source",
            "unknown",
        )
        for record
        in valid_records
    )

    report = {
        "train_records": len(
            train_records
        ),
        "valid_records": len(
            valid_records
        ),
        "total_records": len(
            all_records
        ),
        "positive_records": (
            len(all_records)
            - negative_count
        ),
        "negative_records": (
            negative_count
        ),
        "sources": dict(
            sorted(
                source_counts.items()
            )
        ),
        "train_sources": dict(
            sorted(
                train_source_counts.items()
            )
        ),
        "valid_sources": dict(
            sorted(
                valid_source_counts.items()
            )
        ),
        "build_counters": dict(
            sorted(
                (
                    counters
                    or Counter()
                ).items()
            )
        ),
        "optional_taco": (
            "not included in fast V2 build"
        ),
    }

    (
        output_directory
        / "build_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_dataset_output(
    train_records: list[dict],
    valid_records: list[dict],
    output_directory: Path,
    counters: Counter | None = None,
) -> None:

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_split(
        train_records,
        "train",
        output_directory,
    )

    _write_split(
        valid_records,
        "valid",
        output_directory,
    )

    _write_data_yaml(
        output_directory
    )

    _write_manifest(
        train_records,
        valid_records,
        output_directory,
    )

    _write_build_report(
        train_records,
        valid_records,
        output_directory,
        counters,
    )


def validate_built_dataset(
    output_directory: Path,
) -> dict:

    errors = []

    train_images = (
        output_directory
        / "train"
        / "images"
    )

    train_labels = (
        output_directory
        / "train"
        / "labels"
    )

    valid_images = (
        output_directory
        / "valid"
        / "images"
    )

    valid_labels = (
        output_directory
        / "valid"
        / "labels"
    )

    required = [
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        (
            output_directory
            / "data.yaml"
        ),
        (
            output_directory
            / "manifest.csv"
        ),
        (
            output_directory
            / "build_report.json"
        ),
    ]

    for path in required:

        if not path.exists():

            errors.append(
                f"Missing: "
                f"{path}"
            )

    train_image_files = [
        path
        for path
        in train_images.glob(
            "*"
        )
        if path.is_file()
    ]

    train_label_files = [
        path
        for path
        in train_labels.glob(
            "*.txt"
        )
        if path.is_file()
    ]

    valid_image_files = [
        path
        for path
        in valid_images.glob(
            "*"
        )
        if path.is_file()
    ]

    valid_label_files = [
        path
        for path
        in valid_labels.glob(
            "*.txt"
        )
        if path.is_file()
    ]

    if (
        len(
            train_image_files
        )
        != len(
            train_label_files
        )
    ):

        errors.append(
            "Train image/label "
            "count mismatch"
        )

    if (
        len(
            valid_image_files
        )
        != len(
            valid_label_files
        )
    ):

        errors.append(
            "Valid image/label "
            "count mismatch"
        )

    for labels_directory in (
        train_labels,
        valid_labels,
    ):

        for label_path in (
            labels_directory.glob(
                "*.txt"
            )
        ):

            for line in (
                label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ):

                if not (
                    line.strip()
                ):
                    continue

                try:

                    class_id = int(
                        line.split()[
                            0
                        ]
                    )

                except Exception:

                    errors.append(
                        (
                            "Invalid label: "
                            f"{label_path}"
                        )
                    )

                    continue

                if (
                    class_id
                    not in (
                        0,
                        1,
                        2,
                        3,
                    )
                ):

                    errors.append(
                        (
                            "Invalid class "
                            f"{class_id}: "
                            f"{label_path}"
                        )
                    )

    manifest_path = (
        output_directory
        / "manifest.csv"
    )

    train_families = set()
    valid_families = set()

    if (
        manifest_path.exists()
    ):

        with manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            for row in (
                csv.DictReader(
                    file
                )
            ):

                if (
                    row[
                        "split"
                    ]
                    == "train"
                ):

                    train_families.add(
                        row[
                            "family_id"
                        ]
                    )

                elif (
                    row[
                        "split"
                    ]
                    == "valid"
                ):

                    valid_families.add(
                        row[
                            "family_id"
                        ]
                    )

    family_overlap = (
        train_families
        & valid_families
    )

    if family_overlap:

        errors.append(
            (
                "Family leakage "
                "detected: "
                f"{len(family_overlap)}"
            )
        )

    if errors:

        raise RuntimeError(
            "\n".join(
                errors
            )
        )

    return {
        "train_images": len(
            train_image_files
        ),
        "valid_images": len(
            valid_image_files
        ),
        "family_overlap": 0,
        "classes_valid": True,
    }


def build_waste_v2(
    valid_fraction: float = 0.12,
    max_negative_fraction: float = 0.20,
    seed: int = 26,
) -> dict:

    counters = Counter()

    all_records = []

    print(
        "\n=== Discovering sources ==="
    )

    for (
        source,
        config,
    ) in (
        SOURCE_CONFIGS.items()
    ):

        print(
            f"Reading: "
            f"{source}"
        )

        records = (
            discover_source_records(
                source,
                config,
                counters,
            )
        )

        print(
            (
                "  candidates: "
                f"{len(records)}"
            )
        )

        all_records.extend(
            records
        )

    print(
        "\nInitial candidates:",
        len(
            all_records
        ),
    )

    print(
        (
            "\n=== Benchmark exclusion "
            "and deduplication ==="
        )
    )

    all_records = (
        deduplicate_and_exclude_benchmarks(
            all_records,
            counters,
        )
    )

    print(
        "After dedupe/exclusion:",
        len(
            all_records
        ),
    )

    print(
        "\n=== Controlled negatives ==="
    )

    selected_records = (
        select_controlled_negatives(
            all_records,
            max_negative_fraction=(
                max_negative_fraction
            ),
            seed=seed,
        )
    )

    print(
        "After negative cap:",
        len(
            selected_records
        ),
    )

    print(
        "\n=== Train / valid split ==="
    )

    (
        train_records,
        valid_records,
    ) = (
        split_records_by_family(
            selected_records,
            valid_fraction=(
                valid_fraction
            ),
            seed=seed,
        )
    )

    print(
        "Train:",
        len(
            train_records
        ),
    )

    print(
        "Valid:",
        len(
            valid_records
        ),
    )

    if (
        OUTPUT_DIRECTORY.exists()
    ):

        shutil.rmtree(
            OUTPUT_DIRECTORY
        )

    write_dataset_output(
        train_records,
        valid_records,
        OUTPUT_DIRECTORY,
        counters,
    )

    validation = (
        validate_built_dataset(
            OUTPUT_DIRECTORY
        )
    )

    return validation


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build Smart Recycle "
            "Waste V2 YOLO dataset."
        )
    )

    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--max-negative-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=26,
    )

    arguments = (
        parser.parse_args()
    )

    result = build_waste_v2(
        valid_fraction=(
            arguments.valid_fraction
        ),
        max_negative_fraction=(
            arguments.max_negative_fraction
        ),
        seed=(
            arguments.seed
        ),
    )

    print(
        (
            "\n=== WASTE V2 "
            "BUILD COMPLETE ==="
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":
    main()