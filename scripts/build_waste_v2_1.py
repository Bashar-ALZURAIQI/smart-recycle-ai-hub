from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    root: Path | None
    source_type: str
    class_map: dict[int, int]
    unsupported_class_ids: set[int]
    preserve_empty_labels: bool
    reject_non_detection_labels: bool
    exclude_if_any_unsupported: bool


@dataclass(frozen=True)
class BuildConfig:
    version: int
    seed: int
    valid_fraction: float
    max_negative_fraction: float
    perceptual_dedupe_mode: str
    protected_benchmark_roots: list[Path]
    source_priority: list[str]
    sources: dict[str, SourcePolicy]


@dataclass(frozen=True)
class CandidateRecord:
    source: str
    source_split: str
    image_path: Path
    label_path: Path
    labels: list[str]
    is_negative: bool


@dataclass(frozen=True)
class ImageFingerprint:
    sha256: str
    dhash: str
    width: int
    height: int


@dataclass(frozen=True)
class DeduplicationResult:
    records: list[CandidateRecord]
    exact_removed: int
    perceptual_removed: int


@dataclass(frozen=True)
class BenchmarkProtectionResult:
    records: list[CandidateRecord]
    exact_removed: int
    perceptual_removed: int


@dataclass(frozen=True)
class NegativeLimitResult:
    records: list[CandidateRecord]
    negatives_kept: int
    negatives_removed: int


@dataclass(frozen=True)
class FamilySplitResult:
    train_records: list[CandidateRecord]
    valid_records: list[CandidateRecord]


def load_build_config(
    config_path: Path,
) -> BuildConfig:
    config_path = Path(
        config_path
    ).resolve()

    project_root = (
        config_path.parent.parent
    )

    raw_config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    sources: dict[
        str,
        SourcePolicy,
    ] = {}

    for (
        source_id,
        source_config,
    ) in raw_config[
        "sources"
    ].items():
        root_value = source_config.get(
            "root"
        )

        root = (
            project_root
            / Path(root_value)
            if root_value is not None
            else None
        )

        class_map = {
            int(source_class): int(
                target_class
            )
            for (
                source_class,
                target_class,
            ) in source_config[
                "class_map"
            ].items()
        }

        unsupported_class_ids = {
            int(class_id)
            for class_id in source_config[
                "unsupported_class_ids"
            ]
        }

        sources[
            source_id
        ] = SourcePolicy(
            source_id=source_id,
            root=root,
            source_type=source_config[
                "type"
            ],
            class_map=class_map,
            unsupported_class_ids=(
                unsupported_class_ids
            ),
            preserve_empty_labels=bool(
                source_config[
                    "preserve_empty_labels"
                ]
            ),
            reject_non_detection_labels=bool(
                source_config[
                    "reject_non_detection_labels"
                ]
            ),
            exclude_if_any_unsupported=bool(
                source_config[
                    "exclude_if_any_unsupported"
                ]
            ),
        )

    protected_benchmark_roots = [
        project_root
        / Path(root)
        for root in raw_config[
            "protected_benchmarks"
        ]
    ]

    return BuildConfig(
        version=int(
            raw_config["version"]
        ),
        seed=int(
            raw_config["seed"]
        ),
        valid_fraction=float(
            raw_config[
                "valid_fraction"
            ]
        ),
        max_negative_fraction=float(
            raw_config[
                "max_negative_fraction"
            ]
        ),
        perceptual_dedupe_mode=str(
            raw_config[
                "perceptual_dedupe"
            ][
                "mode"
            ]
        ),
        protected_benchmark_roots=(
            protected_benchmark_roots
        ),
        source_priority=[
            str(source_id)
            for source_id in raw_config[
                "source_priority"
            ]
        ],
        sources=sources,
    )


def family_id_for_image(
    source_id: str,
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

    return f"{source_id}:{stem}"


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

    get_pixels = getattr(
        resized,
        "get_flattened_data",
        resized.getdata,
    )

    pixels = list(
        get_pixels()
    )

    bits: list[int] = []

    for row in range(8):
        row_start = row * 9

        for column in range(8):
            left = pixels[
                row_start + column
            ]

            right = pixels[
                row_start + column + 1
            ]

            bits.append(
                1
                if left > right
                else 0
            )

    value = 0

    for bit in bits:
        value = (
            value << 1
        ) | bit

    return f"{value:016x}"


def image_fingerprint(
    image_path: Path,
) -> ImageFingerprint:
    image_path = Path(
        image_path
    )

    sha256 = hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()

    with Image.open(
        image_path
    ) as image:
        width, height = image.size

        dhash = compute_dhash(
            image
        )

    return ImageFingerprint(
        sha256=sha256,
        dhash=dhash,
        width=width,
        height=height,
    )


def output_filename_for_record(
    record: CandidateRecord,
) -> str:
    fingerprint = image_fingerprint(
        record.image_path
    )

    extension = (
        record.image_path.suffix.lower()
    )

    return (
        f"{record.source}"
        f"__{fingerprint.sha256}"
        f"{extension}"
    )


def write_split_records(
    records: list[CandidateRecord],
    output_root: Path,
    split_name: str,
) -> None:
    output_root = Path(
        output_root
    )

    images_dir = (
        output_root
        / split_name
        / "images"
    )

    labels_dir = (
        output_root
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

    for record in records:
        output_name = (
            output_filename_for_record(
                record
            )
        )

        output_image = (
            images_dir
            / output_name
        )

        output_label = (
            labels_dir
            / Path(
                output_name
            ).with_suffix(
                ".txt"
            ).name
        )

        shutil.copyfile(
            record.image_path,
            output_image,
        )

        label_text = ""

        if record.labels:
            label_text = (
                "\n".join(
                    record.labels
                )
                + "\n"
            )

        output_label.write_text(
            label_text,
            encoding="utf-8",
        )


def write_data_yaml(
    output_root: Path,
) -> None:
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = (
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        "names:\n"
        "  0: plastic\n"
        "  1: metal\n"
        "  2: glass\n"
        "  3: paper_cardboard\n"
    )

    (
        output_root
        / "data.yaml"
    ).write_text(
        content,
        encoding="utf-8",
    )


def write_manifest(
    train_records: list[CandidateRecord],
    valid_records: list[CandidateRecord],
    output_root: Path,
) -> None:
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_root
        / "manifest.csv"
    )

    fieldnames = [
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
    ]

    rows: list[
        dict[str, object]
    ] = []

    for (
        final_split,
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
            fingerprint = image_fingerprint(
                record.image_path
            )

            output_name = (
                output_filename_for_record(
                    record
                )
            )

            rows.append(
                {
                    "source": (
                        record.source
                    ),
                    "source_split": (
                        record.source_split
                    ),
                    "family_id": (
                        family_id_for_image(
                            record.source,
                            record.image_path,
                        )
                    ),
                    "is_negative": (
                        "true"
                        if record.is_negative
                        else "false"
                    ),
                    "source_image": (
                        record.image_path.name
                    ),
                    "output_name": (
                        output_name
                    ),
                    "final_split": (
                        final_split
                    ),
                    "sha256": (
                        fingerprint.sha256
                    ),
                    "dhash": (
                        fingerprint.dhash
                    ),
                    "width": (
                        fingerprint.width
                    ),
                    "height": (
                        fingerprint.height
                    ),
                }
            )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_build_report(
    output_root: Path,
    candidate_count: int,
    benchmark_exact_removed: int,
    benchmark_perceptual_removed: int,
    dedup_exact_removed: int,
    dedup_perceptual_removed: int,
    negatives_kept: int,
    negatives_removed: int,
    train_records: list[CandidateRecord],
    valid_records: list[CandidateRecord],
    invariant_failures: list[str] | None = None,
) -> None:
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if invariant_failures is None:
        invariant_failures = []

    final_records = (
        train_records
        + valid_records
    )

    final_sources: dict[
        str,
        int,
    ] = {}

    final_class_annotations = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
    }

    for record in final_records:
        final_sources[
            record.source
        ] = (
            final_sources.get(
                record.source,
                0,
            )
            + 1
        )

        for label in record.labels:
            parts = label.split()

            if not parts:
                continue

            class_id = int(
                parts[0]
            )

            class_key = str(
                class_id
            )

            if (
                class_key
                not in final_class_annotations
            ):
                raise ValueError(
                    "Unexpected target class "
                    f"in final record: "
                    f"{class_id}"
                )

            final_class_annotations[
                class_key
            ] += 1

    report = {
        "version": "2.1",
        "classes": {
            "0": "plastic",
            "1": "metal",
            "2": "glass",
            "3": "paper_cardboard",
        },
        "counts": {
            "candidate_records": (
                candidate_count
            ),
            "final_records": len(
                final_records
            ),
            "train_records": len(
                train_records
            ),
            "valid_records": len(
                valid_records
            ),
        },
        "benchmark_removals": {
            "exact": (
                benchmark_exact_removed
            ),
            "perceptual": (
                benchmark_perceptual_removed
            ),
        },
        "dedup_removals": {
            "exact": (
                dedup_exact_removed
            ),
            "perceptual": (
                dedup_perceptual_removed
            ),
        },
        "negatives": {
            "kept": (
                negatives_kept
            ),
            "removed": (
                negatives_removed
            ),
        },
        "final_sources": dict(
            sorted(
                final_sources.items()
            )
        ),
        "final_class_annotations": (
            final_class_annotations
        ),
        "invariant_failures": list(
            invariant_failures
        ),
    }

    report_text = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    (
        output_root
        / "build_report.json"
    ).write_text(
        report_text,
        encoding="utf-8",
    )


def collect_benchmark_fingerprints(
    benchmark_roots: list[Path],
) -> list[ImageFingerprint]:
    fingerprints: list[
        ImageFingerprint
    ] = []

    for benchmark_root in benchmark_roots:
        benchmark_root = Path(
            benchmark_root
        )

        for image_path in sorted(
            benchmark_root.rglob("*")
        ):
            if not image_path.is_file():
                continue

            if (
                image_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            fingerprints.append(
                image_fingerprint(
                    image_path
                )
            )

    return fingerprints


def benchmark_match_kind(
    candidate_fingerprint: ImageFingerprint,
    benchmark_fingerprints: list[ImageFingerprint],
) -> str | None:
    for benchmark in benchmark_fingerprints:
        if (
            candidate_fingerprint.sha256
            == benchmark.sha256
        ):
            return "exact"

    for benchmark in benchmark_fingerprints:
        if (
            candidate_fingerprint.dhash
            == benchmark.dhash
            and candidate_fingerprint.width
            == benchmark.width
            and candidate_fingerprint.height
            == benchmark.height
        ):
            return "perceptual_exact"

    return None


def protect_records_from_benchmarks(
    records: list[CandidateRecord],
    benchmark_fingerprints: list[ImageFingerprint],
) -> BenchmarkProtectionResult:
    kept_records: list[
        CandidateRecord
    ] = []

    exact_removed = 0
    perceptual_removed = 0

    for record in records:
        fingerprint = image_fingerprint(
            record.image_path
        )

        match_kind = benchmark_match_kind(
            fingerprint,
            benchmark_fingerprints,
        )

        if match_kind == "exact":
            exact_removed += 1
            continue

        if match_kind == "perceptual_exact":
            perceptual_removed += 1
            continue

        kept_records.append(
            record
        )

    return BenchmarkProtectionResult(
        records=kept_records,
        exact_removed=exact_removed,
        perceptual_removed=perceptual_removed,
    )


def deduplicate_records(
    records: list[CandidateRecord],
    source_priority: list[str],
) -> DeduplicationResult:
    priority_by_source = {
        source_id: index
        for index, source_id in enumerate(
            source_priority
        )
    }

    for record in records:
        if (
            record.source
            not in priority_by_source
        ):
            raise ValueError(
                "Source is missing from source_priority: "
                f"{record.source!r}"
            )

    ordered_records = sorted(
        records,
        key=lambda record: (
            priority_by_source[
                record.source
            ],
            record.source,
            record.image_path.as_posix(),
        ),
    )

    exact_unique_records: list[
        tuple[
            CandidateRecord,
            ImageFingerprint,
        ]
    ] = []

    seen_sha256: set[str] = set()
    exact_removed = 0

    for record in ordered_records:
        fingerprint = image_fingerprint(
            record.image_path
        )

        if (
            fingerprint.sha256
            in seen_sha256
        ):
            exact_removed += 1
            continue

        seen_sha256.add(
            fingerprint.sha256
        )

        exact_unique_records.append(
            (
                record,
                fingerprint,
            )
        )

    kept_records: list[
        CandidateRecord
    ] = []

    seen_perceptual: set[
        tuple[
            str,
            int,
            int,
        ]
    ] = set()

    perceptual_removed = 0

    for (
        record,
        fingerprint,
    ) in exact_unique_records:
        perceptual_key = (
            fingerprint.dhash,
            fingerprint.width,
            fingerprint.height,
        )

        if (
            perceptual_key
            in seen_perceptual
        ):
            perceptual_removed += 1
            continue

        seen_perceptual.add(
            perceptual_key
        )

        kept_records.append(
            record
        )

    return DeduplicationResult(
        records=kept_records,
        exact_removed=exact_removed,
        perceptual_removed=perceptual_removed,
    )


def limit_negative_fraction(
    records: list[CandidateRecord],
    max_negative_fraction: float,
    seed: int,
) -> NegativeLimitResult:
    if not (
        0.0
        <= max_negative_fraction
        < 1.0
    ):
        raise ValueError(
            "max_negative_fraction must be "
            "greater than or equal to 0 "
            "and less than 1."
        )

    positives = [
        record
        for record in records
        if not record.is_negative
    ]

    negatives = [
        record
        for record in records
        if record.is_negative
    ]

    max_negatives = int(
        (
            max_negative_fraction
            * len(positives)
        )
        / (
            1.0
            - max_negative_fraction
        )
    )

    negatives_to_keep = min(
        len(negatives),
        max_negatives,
    )

    ordered_negatives = sorted(
        negatives,
        key=lambda record: (
            record.source,
            record.image_path.as_posix(),
        ),
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        ordered_negatives
    )

    kept_negatives = (
        ordered_negatives[
            :negatives_to_keep
        ]
    )

    kept_records = (
        positives
        + kept_negatives
    )

    return NegativeLimitResult(
        records=kept_records,
        negatives_kept=len(
            kept_negatives
        ),
        negatives_removed=(
            len(negatives)
            - len(kept_negatives)
        ),
    )


def family_safe_split(
    records: list[CandidateRecord],
    valid_fraction: float,
    seed: int,
) -> FamilySplitResult:
    if not (
        0.0
        < valid_fraction
        < 1.0
    ):
        raise ValueError(
            "valid_fraction must be "
            "greater than 0 "
            "and less than 1."
        )

    families: dict[
        str,
        list[CandidateRecord],
    ] = {}

    for record in records:
        family_id = family_id_for_image(
            record.source,
            record.image_path,
        )

        families.setdefault(
            family_id,
            [],
        ).append(
            record
        )

    if len(families) <= 1:
        return FamilySplitResult(
            train_records=list(
                records
            ),
            valid_records=[],
        )

    ordered_family_ids = sorted(
        families
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        ordered_family_ids
    )

    target_valid_records = max(
        1,
        round(
            len(records)
            * valid_fraction
        ),
    )

    selected_valid_families: set[
        str
    ] = set()

    valid_record_count = 0

    source_family_ids: dict[
        str,
        list[str],
    ] = {}

    for family_id in ordered_family_ids:
        family_records = families[
            family_id
        ]

        source = family_records[
            0
        ].source

        source_family_ids.setdefault(
            source,
            [],
        ).append(
            family_id
        )

    source_ids = sorted(
        source_family_ids
    )

    can_represent_all_sources = (
        target_valid_records
        >= len(source_ids)
        and len(families)
        > len(source_ids)
    )

    if can_represent_all_sources:
        for source in source_ids:
            family_id = (
                source_family_ids[
                    source
                ][0]
            )

            selected_valid_families.add(
                family_id
            )

            valid_record_count += len(
                families[
                    family_id
                ]
            )

    for family_id in ordered_family_ids:
        if (
            valid_record_count
            >= target_valid_records
        ):
            break

        if (
            family_id
            in selected_valid_families
        ):
            continue

        if (
            len(
                selected_valid_families
            )
            >= len(families) - 1
        ):
            break

        selected_valid_families.add(
            family_id
        )

        valid_record_count += len(
            families[
                family_id
            ]
        )

    train_records: list[
        CandidateRecord
    ] = []

    valid_records: list[
        CandidateRecord
    ] = []

    for family_id in ordered_family_ids:
        family_records = families[
            family_id
        ]

        if (
            family_id
            in selected_valid_families
        ):
            valid_records.extend(
                family_records
            )
        else:
            train_records.extend(
                family_records
            )

    if not train_records:
        last_valid_family = next(
            reversed(
                [
                    family_id
                    for family_id
                    in ordered_family_ids
                    if (
                        family_id
                        in selected_valid_families
                    )
                ]
            )
        )

        moved_records = families[
            last_valid_family
        ]

        valid_records = [
            record
            for record in valid_records
            if (
                family_id_for_image(
                    record.source,
                    record.image_path,
                )
                != last_valid_family
            )
        ]

        train_records.extend(
            moved_records
        )

    return FamilySplitResult(
        train_records=train_records,
        valid_records=valid_records,
    )


def remap_image_labels(
    lines: list[str],
    policy: SourcePolicy,
) -> tuple[
    list[str],
    str | None,
]:
    normalized_lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    if not normalized_lines:
        if (
            policy.preserve_empty_labels
        ):
            return [], None

        return [], "empty_label"

    parsed: list[
        tuple[
            int,
            float,
            float,
            float,
            float,
        ]
    ] = []

    for line in normalized_lines:
        parts = line.split()

        if len(parts) != 5:
            if (
                policy.reject_non_detection_labels
            ):
                return (
                    [],
                    "non_detection_label",
                )

            raise ValueError(
                "Expected 5 YOLO tokens, "
                f"got {len(parts)}: "
                f"{line!r}"
            )

        source_class = int(
            parts[0]
        )

        x_center = float(
            parts[1]
        )

        y_center = float(
            parts[2]
        )

        width = float(
            parts[3]
        )

        height = float(
            parts[4]
        )

        parsed.append(
            (
                source_class,
                x_center,
                y_center,
                width,
                height,
            )
        )

    if (
        policy.exclude_if_any_unsupported
    ):
        for (
            source_class,
            *_,
        ) in parsed:
            if (
                source_class
                in policy.unsupported_class_ids
            ):
                return (
                    [],
                    "unsupported_class",
                )

    remapped: list[str] = []

    for (
        source_class,
        x_center,
        y_center,
        width,
        height,
    ) in parsed:
        target_class = (
            policy.class_map[
                source_class
            ]
        )

        remapped.append(
            f"{target_class} "
            f"{x_center:.8f} "
            f"{y_center:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

    return remapped, None


def discover_source_records(
    policy: SourcePolicy,
) -> list[CandidateRecord]:
    if policy.root is None:
        raise ValueError(
            f"Source "
            f"{policy.source_id!r} "
            "has no root directory."
        )

    records: list[
        CandidateRecord
    ] = []

    for split_name in (
        "train",
        "valid",
        "test",
    ):
        images_dir = (
            policy.root
            / split_name
            / "images"
        )

        labels_dir = (
            policy.root
            / split_name
            / "labels"
        )

        images_exists = (
            images_dir.exists()
        )

        labels_exists = (
            labels_dir.exists()
        )

        if (
            not images_exists
            and not labels_exists
        ):
            continue

        if (
            images_exists
            != labels_exists
        ):
            raise ValueError(
                "Image/label mismatch "
                f"for source="
                f"{policy.source_id!r}, "
                f"split="
                f"{split_name!r}: "
                f"images_dir_exists="
                f"{images_exists}, "
                f"labels_dir_exists="
                f"{labels_exists}"
            )

        images_by_stem = {
            path.stem: path
            for path in sorted(
                images_dir.iterdir()
            )
            if (
                path.is_file()
                and (
                    path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )
        }

        labels_by_stem = {
            path.stem: path
            for path in sorted(
                labels_dir.iterdir()
            )
            if (
                path.is_file()
                and (
                    path.suffix.lower()
                    == ".txt"
                )
            )
        }

        image_stems = set(
            images_by_stem
        )

        label_stems = set(
            labels_by_stem
        )

        images_without_labels = sorted(
            image_stems
            - label_stems
        )

        labels_without_images = sorted(
            label_stems
            - image_stems
        )

        if (
            images_without_labels
            or labels_without_images
        ):
            raise ValueError(
                "Image/label mismatch "
                f"for source="
                f"{policy.source_id!r}, "
                f"split="
                f"{split_name!r}: "
                f"images_without_labels="
                f"{images_without_labels[:10]}, "
                f"labels_without_images="
                f"{labels_without_images[:10]}"
            )

        paired_stems = sorted(
            image_stems
        )

        for stem in paired_stems:
            image_path = (
                images_by_stem[
                    stem
                ]
            )

            label_path = (
                labels_by_stem[
                    stem
                ]
            )

            raw_lines = (
                label_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            )

            (
                labels,
                reason,
            ) = remap_image_labels(
                raw_lines,
                policy,
            )

            if reason is not None:
                continue

            records.append(
                CandidateRecord(
                    source=(
                        policy.source_id
                    ),
                    source_split=(
                        split_name
                    ),
                    image_path=(
                        image_path
                    ),
                    label_path=(
                        label_path
                    ),
                    labels=labels,
                    is_negative=(
                        not labels
                    ),
                )
            )

    return records


def validate_final_dataset(
    train_records: list[CandidateRecord],
    valid_records: list[CandidateRecord],
    benchmark_fingerprints: list[ImageFingerprint],
    output_root: Path,
) -> list[str]:
    failures: list[str] = []

    final_records = (
        train_records
        + valid_records
    )

    for record in final_records:
        for label in record.labels:
            parts = label.split()

            if len(parts) != 5:
                failures.append(
                    "non_five_token_label"
                )
                continue

            try:
                class_id = int(
                    parts[0]
                )

                x_center = float(
                    parts[1]
                )

                y_center = float(
                    parts[2]
                )

                width = float(
                    parts[3]
                )

                height = float(
                    parts[4]
                )
            except ValueError:
                failures.append(
                    "invalid_label_value"
                )
                continue

            if class_id not in {
                0,
                1,
                2,
                3,
            }:
                failures.append(
                    "invalid_class_id"
                )

            if not (
                0.0
                <= x_center
                <= 1.0
            ):
                failures.append(
                    "invalid_x_center"
                )

            if not (
                0.0
                <= y_center
                <= 1.0
            ):
                failures.append(
                    "invalid_y_center"
                )

            if not (
                0.0
                < width
                <= 1.0
            ):
                failures.append(
                    "invalid_width"
                )

            if not (
                0.0
                < height
                <= 1.0
            ):
                failures.append(
                    "invalid_height"
                )

    train_families = {
        family_id_for_image(
            record.source,
            record.image_path,
        )
        for record in train_records
    }

    valid_families = {
        family_id_for_image(
            record.source,
            record.image_path,
        )
        for record in valid_records
    }

    if not train_families.isdisjoint(
        valid_families
    ):
        failures.append(
            "family_overlap"
        )

    train_sha256: set[str] = set()

    train_perceptual: set[
        tuple[
            str,
            int,
            int,
        ]
    ] = set()

    for record in train_records:
        fingerprint = image_fingerprint(
            record.image_path
        )

        train_sha256.add(
            fingerprint.sha256
        )

        train_perceptual.add(
            (
                fingerprint.dhash,
                fingerprint.width,
                fingerprint.height,
            )
        )

    for record in valid_records:
        fingerprint = image_fingerprint(
            record.image_path
        )

        if (
            fingerprint.sha256
            in train_sha256
        ):
            failures.append(
                "exact_train_valid_overlap"
            )

        perceptual_key = (
            fingerprint.dhash,
            fingerprint.width,
            fingerprint.height,
        )

        if (
            perceptual_key
            in train_perceptual
        ):
            failures.append(
                "perceptual_train_valid_overlap"
            )

    for record in final_records:
        fingerprint = image_fingerprint(
            record.image_path
        )

        match_kind = benchmark_match_kind(
            fingerprint,
            benchmark_fingerprints,
        )

        if match_kind == "exact":
            failures.append(
                "benchmark_exact_overlap"
            )

        if (
            match_kind
            == "perceptual_exact"
        ):
            failures.append(
                "benchmark_perceptual_overlap"
            )

    output_root = Path(
        output_root
    )

    for split_name in (
        "train",
        "valid",
    ):
        images_dir = (
            output_root
            / split_name
            / "images"
        )

        labels_dir = (
            output_root
            / split_name
            / "labels"
        )

        image_stems = {
            path.stem
            for path in images_dir.iterdir()
            if (
                path.is_file()
                and (
                    path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )
        }

        label_stems = {
            path.stem
            for path in labels_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                == ".txt"
            )
        }

        if (
            image_stems
            != label_stems
        ):
            failures.append(
                f"{split_name}_image_label_mismatch"
            )

        for label_path in sorted(
            labels_dir.glob(
                "*.txt"
            )
        ):
            for raw_line in (
                label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ):
                line = raw_line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    failures.append(
                        "output_non_five_token_label"
                    )
                    continue

                try:
                    class_id = int(
                        parts[0]
                    )

                    x_center = float(
                        parts[1]
                    )

                    y_center = float(
                        parts[2]
                    )

                    width = float(
                        parts[3]
                    )

                    height = float(
                        parts[4]
                    )
                except ValueError:
                    failures.append(
                        "output_invalid_label_value"
                    )
                    continue

                if class_id not in {
                    0,
                    1,
                    2,
                    3,
                }:
                    failures.append(
                        "output_invalid_class_id"
                    )

                if not (
                    0.0
                    <= x_center
                    <= 1.0
                ):
                    failures.append(
                        "output_invalid_x_center"
                    )

                if not (
                    0.0
                    <= y_center
                    <= 1.0
                ):
                    failures.append(
                        "output_invalid_y_center"
                    )

                if not (
                    0.0
                    < width
                    <= 1.0
                ):
                    failures.append(
                        "output_invalid_width"
                    )

                if not (
                    0.0
                    < height
                    <= 1.0
                ):
                    failures.append(
                        "output_invalid_height"
                    )

    return sorted(
        set(
            failures
        )
    )


def build_waste_v2_1(
    config_path: Path,
    output_root: Path,
) -> None:
    config = load_build_config(
        config_path
    )

    if (
        config.perceptual_dedupe_mode
        != "exact_dhash_dimensions"
    ):
        raise ValueError(
            "Unsupported perceptual_dedupe mode: "
            f"{config.perceptual_dedupe_mode!r}"
        )

    candidate_records: list[
        CandidateRecord
    ] = []

    for source_id in (
        config.source_priority
    ):
        if source_id not in (
            config.sources
        ):
            raise ValueError(
                "Source in source_priority "
                "is missing from sources: "
                f"{source_id!r}"
            )

        policy = config.sources[
            source_id
        ]

        source_records = (
            discover_source_records(
                policy
            )
        )

        candidate_records.extend(
            source_records
        )

    if not candidate_records:
        raise ValueError(
            "No candidate records were discovered."
        )

    negative_result = (
        limit_negative_fraction(
            candidate_records,
            max_negative_fraction=(
                config.max_negative_fraction
            ),
            seed=config.seed,
        )
    )

    benchmark_fingerprints = (
        collect_benchmark_fingerprints(
            config.protected_benchmark_roots
        )
    )

    benchmark_result = (
        protect_records_from_benchmarks(
            negative_result.records,
            benchmark_fingerprints,
        )
    )

    deduplication_result = (
        deduplicate_records(
            benchmark_result.records,
            source_priority=(
                config.source_priority
            ),
        )
    )

    split_result = (
        family_safe_split(
            deduplication_result.records,
            valid_fraction=(
                config.valid_fraction
            ),
            seed=config.seed,
        )
    )

    output_root = Path(
        output_root
    )

    if output_root.exists():
        shutil.rmtree(
            output_root
        )

    write_split_records(
        split_result.train_records,
        output_root=output_root,
        split_name="train",
    )

    write_split_records(
        split_result.valid_records,
        output_root=output_root,
        split_name="valid",
    )

    write_data_yaml(
        output_root
    )

    write_manifest(
        train_records=(
            split_result.train_records
        ),
        valid_records=(
            split_result.valid_records
        ),
        output_root=output_root,
    )

    invariant_failures = (
        validate_final_dataset(
            train_records=(
                split_result.train_records
            ),
            valid_records=(
                split_result.valid_records
            ),
            benchmark_fingerprints=(
                benchmark_fingerprints
            ),
            output_root=output_root,
        )
    )

    write_build_report(
        output_root=output_root,
        candidate_count=len(
            candidate_records
        ),
        benchmark_exact_removed=(
            benchmark_result.exact_removed
        ),
        benchmark_perceptual_removed=(
            benchmark_result.perceptual_removed
        ),
        dedup_exact_removed=(
            deduplication_result.exact_removed
        ),
        dedup_perceptual_removed=(
            deduplication_result.perceptual_removed
        ),
        negatives_kept=(
            negative_result.negatives_kept
        ),
        negatives_removed=(
            negative_result.negatives_removed
        ),
        train_records=(
            split_result.train_records
        ),
        valid_records=(
            split_result.valid_records
        ),
        invariant_failures=(
            invariant_failures
        ),
    )

    if invariant_failures:
        raise ValueError(
            "Waste V2.1 final validation "
            "failed: "
            + ", ".join(
                invariant_failures
            )
        )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the reproducible "
            "Waste V2.1 YOLO dataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/waste_v2_1_mapping.yaml"
        ),
        help=(
            "Path to the Waste V2.1 "
            "mapping configuration."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/waste_v2_1"
        ),
        help=(
            "Destination directory for "
            "the generated Waste V2.1 dataset."
        ),
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(
        argv
    )

    build_waste_v2_1(
        config_path=args.config,
        output_root=args.output_root,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )