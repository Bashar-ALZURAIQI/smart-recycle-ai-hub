from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}


def find_images(root: Path) -> list[Path]:
    """
    Find supported image files recursively below root.
    """
    root = Path(root)

    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_class_id(
    raw_value: str,
    nc: int,
) -> int:
    """
    Parse and validate a YOLO class ID.
    """
    try:
        class_value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Class ID is not numeric: {raw_value}"
        ) from exc

    if not class_value.is_integer():
        raise ValueError(
            f"Class ID must be an integer, got {raw_value}"
        )

    class_id = int(class_value)

    if not 0 <= class_id < nc:
        raise ValueError(
            f"Class ID {class_id} is outside 0..{nc - 1}"
        )

    return class_id


def validate_normalized_coordinate(
    value: float,
    name: str,
) -> float:
    """
    Validate one normalized coordinate.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be normalized, got {value}"
        )

    return value


def validate_yolo_line(
    line: str,
    nc: int,
) -> tuple[int, float, float, float, float]:
    """
    Parse and validate one YOLO detection bounding-box line.

    Expected format:
        class_id x_center y_center width height
    """
    parts = line.strip().split()

    if len(parts) != 5:
        raise ValueError(
            f"Expected 5 YOLO values, got {len(parts)}"
        )

    class_id = validate_class_id(
        parts[0],
        nc=nc,
    )

    try:
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError as exc:
        raise ValueError(
            "YOLO annotation contains non-numeric values"
        ) from exc

    validate_normalized_coordinate(
        x_center,
        "x_center",
    )

    validate_normalized_coordinate(
        y_center,
        "y_center",
    )

    if not 0.0 < width <= 1.0:
        raise ValueError(
            f"width must be in (0, 1], got {width}"
        )

    if not 0.0 < height <= 1.0:
        raise ValueError(
            f"height must be in (0, 1], got {height}"
        )

    return (
        class_id,
        x_center,
        y_center,
        width,
        height,
    )


def parse_yolo_annotation(
    line: str,
    nc: int,
) -> dict:
    """
    Parse one YOLO annotation.

    Supported formats:

    Detection:
        class_id x_center y_center width height

    Segmentation:
        class_id x1 y1 x2 y2 x3 y3 ...

    A segmentation polygon requires at least three coordinate pairs.
    """
    parts = line.strip().split()

    if len(parts) == 5:
        (
            class_id,
            x_center,
            y_center,
            width,
            height,
        ) = validate_yolo_line(
            line,
            nc=nc,
        )

        return {
            "kind": "bbox",
            "class_id": class_id,
            "bbox": (
                x_center,
                y_center,
                width,
                height,
            ),
        }

    if len(parts) < 7:
        raise ValueError(
            "YOLO segmentation requires at least "
            "3 coordinate pairs"
        )

    if len(parts) % 2 == 0:
        raise ValueError(
            "YOLO segmentation must contain "
            "1 class ID plus x/y coordinate pairs"
        )

    class_id = validate_class_id(
        parts[0],
        nc=nc,
    )

    try:
        coordinates = [
            float(value)
            for value in parts[1:]
        ]
    except ValueError as exc:
        raise ValueError(
            "YOLO segmentation contains non-numeric values"
        ) from exc

    points = []

    for index in range(
        0,
        len(coordinates),
        2,
    ):
        x = coordinates[index]
        y = coordinates[index + 1]

        validate_normalized_coordinate(
            x,
            "polygon x",
        )

        validate_normalized_coordinate(
            y,
            "polygon y",
        )

        points.append(
            (x, y)
        )

    xs = [
        point[0]
        for point in points
    ]

    ys = [
        point[1]
        for point in points
    ]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    if xmax <= xmin:
        raise ValueError(
            "YOLO segmentation polygon has zero width"
        )

    if ymax <= ymin:
        raise ValueError(
            "YOLO segmentation polygon has zero height"
        )

    return {
        "kind": "segmentation",
        "class_id": class_id,
        "points": points,
        "bounds": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        },
    }


def count_yolo_classes(
    label_paths: Iterable[Path],
    nc: int,
) -> dict:
    """
    Count valid YOLO detection and segmentation annotations.

    Invalid annotations are counted separately.
    """
    class_counts: Counter[int] = Counter()

    valid_annotations = 0
    bbox_annotations = 0
    segmentation_annotations = 0
    invalid_lines = 0
    files_scanned = 0

    for label_path in label_paths:
        label_path = Path(label_path)
        files_scanned += 1

        try:
            text = label_path.read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeError):
            invalid_lines += 1
            continue

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            try:
                parsed = parse_yolo_annotation(
                    line,
                    nc=nc,
                )
            except ValueError:
                invalid_lines += 1
                continue

            class_id = parsed["class_id"]

            class_counts[class_id] += 1
            valid_annotations += 1

            if parsed["kind"] == "bbox":
                bbox_annotations += 1

            elif parsed["kind"] == "segmentation":
                segmentation_annotations += 1

    return {
        "class_counts": dict(
            sorted(class_counts.items())
        ),
        "valid_annotations": valid_annotations,
        "bbox_annotations": bbox_annotations,
        "segmentation_annotations": (
            segmentation_annotations
        ),
        "invalid_lines": invalid_lines,
        "files_scanned": files_scanned,
    }


def infer_split(
    path: Path,
    root: Path,
) -> str:
    """
    Infer train/valid/test split from a file path.

    Supports layouts such as:

        train/images/file.jpg
        train/labels/file.txt

    and:

        images/train/file.jpg
        labels/val/file.txt
    """
    path = Path(path)
    root = Path(root)

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts

    for part in relative_parts:
        normalized = part.lower()

        if normalized in SPLIT_ALIASES:
            return SPLIT_ALIASES[normalized]

    return "unknown"


def find_label_files(root: Path) -> list[Path]:
    """
    Find YOLO .txt label files below directories named 'labels'.
    """
    root = Path(root)

    if not root.exists():
        return []

    label_files = []

    for path in root.rglob("*.txt"):
        if not path.is_file():
            continue

        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts

        if any(
            part.lower() == "labels"
            for part in relative_parts
        ):
            label_files.append(path)

    return sorted(label_files)


def audit_yolo_source(
    root: Path,
    nc: int,
) -> dict:
    """
    Audit one YOLO-format dataset source.
    """
    root = Path(root)

    images = find_images(root)
    label_files = find_label_files(root)

    image_splits: Counter[str] = Counter(
        infer_split(
            image_path,
            root,
        )
        for image_path in images
    )

    label_splits: Counter[str] = Counter(
        infer_split(
            label_path,
            root,
        )
        for label_path in label_files
    )

    annotation_report = count_yolo_classes(
        label_files,
        nc=nc,
    )

    return {
        "images": len(images),
        "label_files": len(label_files),
        "image_splits": dict(
            sorted(image_splits.items())
        ),
        "label_splits": dict(
            sorted(label_splits.items())
        ),
        "class_counts": annotation_report[
            "class_counts"
        ],
        "valid_annotations": annotation_report[
            "valid_annotations"
        ],
        "bbox_annotations": annotation_report[
            "bbox_annotations"
        ],
        "segmentation_annotations": (
            annotation_report[
                "segmentation_annotations"
            ]
        ),
        "invalid_lines": annotation_report[
            "invalid_lines"
        ],
    }


def build_basename_index(
    root: Path,
) -> dict[str, Path]:
    """
    Build an index that maps image basenames to image paths.
    """
    index: dict[str, Path] = {}

    for image_path in find_images(root):
        index.setdefault(
            image_path.name,
            image_path,
        )

    return index


def locate_coco_image(
    root: Path,
    file_name: str,
    basename_index: dict[str, Path],
) -> Path | None:
    """
    Locate an image declared by a COCO JSON record.
    """
    root = Path(root)

    direct_path = root / Path(file_name)

    if direct_path.exists() and direct_path.is_file():
        return direct_path

    return basename_index.get(
        Path(file_name).name
    )


def audit_coco_source(
    root: Path,
    annotations_path: Path,
) -> dict:
    """
    Audit one COCO-format dataset source.
    """
    root = Path(root)
    annotations_path = Path(annotations_path)

    data = json.loads(
        annotations_path.read_text(
            encoding="utf-8"
        )
    )

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_names = {
        int(category["id"]): str(category["name"])
        for category in categories
    }

    category_counts: Counter[int] = Counter()

    for annotation in annotations:
        category_id = int(
            annotation["category_id"]
        )

        category_counts[category_id] += 1

    basename_index = build_basename_index(root)

    images_found = 0

    for image in images:
        file_name = str(
            image.get("file_name", "")
        )

        if not file_name:
            continue

        image_path = locate_coco_image(
            root,
            file_name,
            basename_index,
        )

        if image_path is not None:
            images_found += 1

    images_declared = len(images)

    return {
        "images_declared": images_declared,
        "images_found": images_found,
        "missing_images": (
            images_declared - images_found
        ),
        "annotations": len(annotations),
        "categories": len(categories),
        "category_counts": dict(
            sorted(category_counts.items())
        ),
        "category_names": dict(
            sorted(category_names.items())
        ),
    }


def to_json_compatible(value):
    """
    Normalize an object through JSON so returned data exactly matches
    what will be loaded back from the saved report.
    """
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
        )
    )


def audit_sources(
    source_specs: dict,
    output_path: Path,
) -> dict:
    """
    Audit multiple YOLO and COCO sources and save one unified report.
    """
    output_path = Path(output_path)

    sources_report = {}

    for source_name, spec in source_specs.items():
        source_format = str(
            spec["format"]
        ).lower()

        root = Path(spec["root"])

        if source_format == "yolo":
            result = audit_yolo_source(
                root,
                nc=int(spec["nc"]),
            )

        elif source_format == "coco":
            result = audit_coco_source(
                root,
                Path(spec["annotations"]),
            )

        else:
            raise ValueError(
                f"Unsupported source format: "
                f"{source_format}"
            )

        result["format"] = source_format
        result["root"] = str(root)

        if source_format == "coco":
            result["annotations_path"] = str(
                spec["annotations"]
            )

        sources_report[source_name] = result

    report = {
        "sources": sources_report,
    }

    report = to_json_compatible(report)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def default_source_specs(
    project_root: Path,
) -> dict:
    """
    Return the real Smart Recycle AI-Hub V2 audit sources.
    """
    project_root = Path(project_root)

    return {
        "waste_v1": {
            "format": "yolo",
            "root": (
                project_root
                / "data"
                / "processed"
                / "waste_v1"
            ),
            "nc": 4,
        },

        "taco_optional": {
            "format": "coco",
            "root": (
                project_root
                / "data"
                / "raw"
                / "taco_optional"
            ),
            "annotations": (
                project_root
                / "data"
                / "raw"
                / "taco_optional"
                / "data"
                / "annotations.json"
            ),
        },

        "garbage_classification": {
            "format": "yolo",
            "root": (
                project_root
                / "data"
                / "raw"
                / "external_dataset_2"
                / "GARBAGE CLASSIFICATION"
            ),
            "nc": 6,
        },

        "recyclable_waste_detection": {
            "format": "yolo",
            "root": (
                project_root
                / "data"
                / "raw"
                / "recyclable_waste_detection"
            ),
            "nc": 5,
        },

        "conveyor_waste_belt": {
            "format": "yolo",
            "root": (
                project_root
                / "data"
                / "raw"
                / "conveyor_waste_belt"
            ),
            "nc": 5,
        },

        "external_test_v1": {
            "format": "yolo",
            "root": (
                project_root
                / "data"
                / "benchmarks"
                / "external_test_v1"
            ),
            "nc": 4,
        },

        "reject_challenge_v1": {
            "format": "yolo",
            "root": (
                project_root
                / "data"
                / "benchmarks"
                / "reject_challenge_v1"
            ),
            "nc": 4,
        },
    }


def print_summary(
    report: dict,
) -> None:
    """
    Print a short human-readable audit summary.
    """
    print()
    print("Waste V2 Source Audit")
    print("=" * 60)

    for name, source in report["sources"].items():
        print()
        print(f"[{name}]")
        print(f"format: {source['format']}")

        if source["format"] == "yolo":
            print(
                f"images: "
                f"{source['images']}"
            )
            print(
                f"label files: "
                f"{source['label_files']}"
            )
            print(
                f"valid annotations: "
                f"{source['valid_annotations']}"
            )
            print(
                f"bbox annotations: "
                f"{source['bbox_annotations']}"
            )
            print(
                f"segmentation annotations: "
                f"{source['segmentation_annotations']}"
            )
            print(
                f"invalid lines: "
                f"{source['invalid_lines']}"
            )
            print(
                f"class counts: "
                f"{source['class_counts']}"
            )
            print(
                f"image splits: "
                f"{source['image_splits']}"
            )

        else:
            print(
                f"images declared: "
                f"{source['images_declared']}"
            )
            print(
                f"images found: "
                f"{source['images_found']}"
            )
            print(
                f"missing images: "
                f"{source['missing_images']}"
            )
            print(
                f"annotations: "
                f"{source['annotations']}"
            )
            print(
                f"categories: "
                f"{source['categories']}"
            )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    default_project_root = (
        Path(__file__).resolve().parents[1]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Audit all Smart Recycle AI-Hub "
            "Waste V2 candidate sources."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root,
        help=(
            "Smart Recycle AI-Hub repository root."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Run the real Waste V2 source audit.
    """
    args = parse_args()

    project_root = (
        args.project_root.resolve()
    )

    if args.output is None:
        output_path = (
            project_root
            / "data"
            / "inspection"
            / "waste_v2_source_audit.json"
        )
    else:
        output_path = args.output.resolve()

    specs = default_source_specs(
        project_root
    )

    report = audit_sources(
        specs,
        output_path=output_path,
    )

    print_summary(report)

    print()
    print("=" * 60)
    print("Audit report saved to:")
    print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())