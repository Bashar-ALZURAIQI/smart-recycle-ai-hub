from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


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

        paired_stems = sorted(image_stems)

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