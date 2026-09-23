from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

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


def family_id_for_image(
    source_id: str,
    image_path: Path,
) -> str:
    stem = image_path.stem

    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]

    stem = re.sub(
        r"(?i)[_-]aug(?:mented)?[_-]?\d+$",
        "",
        stem,
    )

    return f"{source_id}:{stem}"


def compute_dhash(
    image: Image.Image,
) -> str:
    grayscale = image.convert("L")

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
    image_path = Path(image_path)

    sha256 = hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()

    with Image.open(image_path) as image:
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


def collect_benchmark_fingerprints(
    benchmark_roots: list[Path],
) -> list[ImageFingerprint]:
    fingerprints: list[ImageFingerprint] = []

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
    kept_records: list[CandidateRecord] = []

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
        if record.source not in priority_by_source:
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
        tuple[CandidateRecord, ImageFingerprint]
    ] = []

    seen_sha256: set[str] = set()
    exact_removed = 0

    for record in ordered_records:
        fingerprint = image_fingerprint(
            record.image_path
        )

        if fingerprint.sha256 in seen_sha256:
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

    kept_records: list[CandidateRecord] = []

    seen_perceptual: set[
        tuple[str, int, int]
    ] = set()

    perceptual_removed = 0

    for record, fingerprint in exact_unique_records:
        perceptual_key = (
            fingerprint.dhash,
            fingerprint.width,
            fingerprint.height,
        )

        if perceptual_key in seen_perceptual:
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


def remap_image_labels(
    lines: list[str],
    policy: SourcePolicy,
) -> tuple[list[str], str | None]:
    normalized_lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    if not normalized_lines:
        if policy.preserve_empty_labels:
            return [], None

        return [], "empty_label"

    parsed: list[tuple[int, float, float, float, float]] = []

    for line in normalized_lines:
        parts = line.split()

        if len(parts) != 5:
            if policy.reject_non_detection_labels:
                return [], "non_detection_label"

            raise ValueError(
                f"Expected 5 YOLO tokens, got {len(parts)}: {line!r}"
            )

        source_class = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        parsed.append(
            (
                source_class,
                x_center,
                y_center,
                width,
                height,
            )
        )

    if policy.exclude_if_any_unsupported:
        for source_class, *_ in parsed:
            if source_class in policy.unsupported_class_ids:
                return [], "unsupported_class"

    remapped: list[str] = []

    for (
        source_class,
        x_center,
        y_center,
        width,
        height,
    ) in parsed:
        target_class = policy.class_map[source_class]

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
            f"Source {policy.source_id!r} has no root directory."
        )

    records: list[CandidateRecord] = []

    for split_name in ("train", "valid", "test"):
        images_dir = policy.root / split_name / "images"
        labels_dir = policy.root / split_name / "labels"

        images_exists = images_dir.exists()
        labels_exists = labels_dir.exists()

        if not images_exists and not labels_exists:
            continue

        if images_exists != labels_exists:
            raise ValueError(
                "Image/label mismatch "
                f"for source={policy.source_id!r}, "
                f"split={split_name!r}: "
                f"images_dir_exists={images_exists}, "
                f"labels_dir_exists={labels_exists}"
            )

        images_by_stem = {
            path.stem: path
            for path in sorted(images_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        labels_by_stem = {
            path.stem: path
            for path in sorted(labels_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() == ".txt"
        }

        image_stems = set(images_by_stem)
        label_stems = set(labels_by_stem)

        images_without_labels = sorted(
            image_stems - label_stems
        )

        labels_without_images = sorted(
            label_stems - image_stems
        )

        if images_without_labels or labels_without_images:
            raise ValueError(
                "Image/label mismatch "
                f"for source={policy.source_id!r}, "
                f"split={split_name!r}: "
                f"images_without_labels="
                f"{images_without_labels[:10]}, "
                f"labels_without_images="
                f"{labels_without_images[:10]}"
            )

        paired_stems = sorted(
            image_stems
        )

        for stem in paired_stems:
            image_path = images_by_stem[stem]
            label_path = labels_by_stem[stem]

            raw_lines = label_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            labels, reason = remap_image_labels(
                raw_lines,
                policy,
            )

            if reason is not None:
                continue

            records.append(
                CandidateRecord(
                    source=policy.source_id,
                    source_split=split_name,
                    image_path=image_path,
                    label_path=label_path,
                    labels=labels,
                    is_negative=not labels,
                )
            )

    return records