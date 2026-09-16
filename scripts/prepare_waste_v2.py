from __future__ import annotations

from audit_waste_v2_sources import parse_yolo_annotation


# ============================================================
# Final Smart Recycle AI-Hub class schema
# ============================================================

FINAL_CLASS_NAMES = {
    0: "plastic",
    1: "metal",
    2: "glass",
    3: "paper_cardboard",
}


# ============================================================
# Waste V1
# ============================================================

WASTE_V1_CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}


# ============================================================
# Garbage Classification
#
# 0 BIODEGRADABLE
# 1 CARDBOARD
# 2 GLASS
# 3 METAL
# 4 PAPER
# 5 PLASTIC
# ============================================================

GARBAGE_CLASSIFICATION_CLASS_MAP = {
    0: None,
    1: 3,
    2: 2,
    3: 1,
    4: 3,
    5: 0,
}


# ============================================================
# Recyclable Waste Detection
#
# 0 Cardboard
# 1 Glass
# 2 Metal
# 3 Paper
# 4 Plastic
# ============================================================

RECYCLABLE_WASTE_DETECTION_CLASS_MAP = {
    0: 3,
    1: 2,
    2: 1,
    3: 3,
    4: 0,
}


# ============================================================
# Conveyor Waste Belt
#
# 0 Glass
# 1 Metal-Other
# 2 Organic Food
# 3 Paper-Cardboard
# 4 Plastic
# ============================================================

CONVEYOR_WASTE_BELT_CLASS_MAP = {
    0: 2,
    1: 1,
    2: None,
    3: 3,
    4: 0,
}


# ============================================================
# TACO
# ============================================================

TACO_PLASTIC_CLASSES = {
    4,
    5,
    7,
    21,
    24,
    27,
    29,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    47,
    48,
    49,
    54,
    55,
    57,
}


TACO_METAL_CLASSES = {
    0,
    8,
    10,
    11,
    12,
    28,
    50,
    52,
}


TACO_GLASS_CLASSES = {
    6,
    9,
    23,
    26,
}


TACO_PAPER_CARDBOARD_CLASSES = {
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    30,
    32,
    33,
    34,
    56,
}


TACO_CLASS_MAP = {
    **{
        class_id: 0
        for class_id in TACO_PLASTIC_CLASSES
    },
    **{
        class_id: 1
        for class_id in TACO_METAL_CLASSES
    },
    **{
        class_id: 2
        for class_id in TACO_GLASS_CLASSES
    },
    **{
        class_id: 3
        for class_id in TACO_PAPER_CARDBOARD_CLASSES
    },
}


# ============================================================
# Unified source mapping registry
# ============================================================

SOURCE_CLASS_MAPS = {
    "waste_v1": WASTE_V1_CLASS_MAP,
    "garbage_classification": (
        GARBAGE_CLASSIFICATION_CLASS_MAP
    ),
    "recyclable_waste_detection": (
        RECYCLABLE_WASTE_DETECTION_CLASS_MAP
    ),
    "conveyor_waste_belt": (
        CONVEYOR_WASTE_BELT_CLASS_MAP
    ),
    "taco_optional": TACO_CLASS_MAP,
}


# ============================================================
# Controlled-negative source classes
#
# These are explicitly known organic / biodegradable classes
# that may later enter a LIMITED negative-training subset.
#
# An unmapped class is NOT automatically considered negative.
# ============================================================

CONTROLLED_NEGATIVE_CLASSES = {
    "waste_v1": set(),

    "garbage_classification": {
        0,  # BIODEGRADABLE
    },

    "recyclable_waste_detection": set(),

    "conveyor_waste_belt": {
        2,  # Organic Food
    },

    "taco_optional": {
        25,  # Food waste
    },
}


def map_source_class(
    source_name: str,
    source_class: int,
) -> int | None:
    """
    Convert one source class ID into the final four-class
    Smart Recycle AI-Hub schema.

    Returns:
        0 -> plastic
        1 -> metal
        2 -> glass
        3 -> paper_cardboard

        None -> the class is intentionally not mapped to
                a final target class.

    Important:
        None does not automatically mean the object is safe
        to use as background.
    """
    if source_name not in SOURCE_CLASS_MAPS:
        raise ValueError(
            f"Unknown Waste V2 source: {source_name}"
        )

    source_class = int(source_class)

    return SOURCE_CLASS_MAPS[
        source_name
    ].get(source_class)


def polygon_to_yolo_bbox(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """
    Convert normalized segmentation polygon points into an
    enclosing normalized YOLO detection bounding box.

    Returns:
        x_center, y_center, width, height
    """
    xs = [
        x
        for x, _ in points
    ]

    ys = [
        y
        for _, y in points
    ]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    x_center = (
        xmin + xmax
    ) / 2.0

    y_center = (
        ymin + ymax
    ) / 2.0

    width = (
        xmax - xmin
    )

    height = (
        ymax - ymin
    )

    return (
        x_center,
        y_center,
        width,
        height,
    )


def convert_yolo_annotation(
    source_name: str,
    line: str,
    nc: int,
) -> tuple[int, float, float, float, float] | None:
    """
    Convert one YOLO source annotation into the final
    Waste V2 detection schema.

    Bounding-box annotations retain their geometry.

    Segmentation polygons are converted to their enclosing
    detection bounding box.

    Returns None when the source class does not map to one
    of the four final target classes.
    """
    parsed = parse_yolo_annotation(
        line,
        nc=nc,
    )

    final_class_id = map_source_class(
        source_name,
        parsed["class_id"],
    )

    if final_class_id is None:
        return None

    if parsed["kind"] == "bbox":
        (
            x_center,
            y_center,
            width,
            height,
        ) = parsed["bbox"]

    elif parsed["kind"] == "segmentation":
        (
            x_center,
            y_center,
            width,
            height,
        ) = polygon_to_yolo_bbox(
            parsed["points"]
        )

    else:
        raise ValueError(
            f"Unsupported YOLO annotation kind: "
            f"{parsed['kind']}"
        )

    return (
        final_class_id,
        x_center,
        y_center,
        width,
        height,
    )


def classify_image_classes(
    source_name: str,
    source_classes: list[int],
) -> str:
    """
    Classify an image using all annotated source classes.

    Returns:
        target
            Every annotated object belongs to one of the
            four final target classes.

        negative_candidate
            Every object belongs only to explicitly approved
            organic / biodegradable negative classes.

        ambiguous
            Target and non-target objects are mixed, or an
            unsupported non-target class exists.

        empty
            No annotations were supplied.
    """
    if not source_classes:
        return "empty"

    if source_name not in SOURCE_CLASS_MAPS:
        raise ValueError(
            f"Unknown Waste V2 source: {source_name}"
        )

    target_found = False
    negative_found = False
    unsupported_found = False

    safe_negative_classes = (
        CONTROLLED_NEGATIVE_CLASSES[
            source_name
        ]
    )

    for source_class in source_classes:
        source_class = int(source_class)

        final_class = map_source_class(
            source_name,
            source_class,
        )

        if final_class is not None:
            target_found = True
            continue

        if source_class in safe_negative_classes:
            negative_found = True
            continue

        unsupported_found = True

    if (
        target_found
        and not negative_found
        and not unsupported_found
    ):
        return "target"

    if (
        negative_found
        and not target_found
        and not unsupported_found
    ):
        return "negative_candidate"

    return "ambiguous"


def prepare_yolo_image_annotations(
    source_name: str,
    lines: list[str],
    nc: int,
) -> dict:
    """
    Prepare all YOLO annotations belonging to one image.

    The function makes the image-level decision BEFORE
    generating final training annotations.

    Result format:

        {
            "status": "target",
            "annotations": [
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                ),
                ...
            ],
        }

    Image-level rules:

        target:
            All source objects map cleanly to target classes.
            Final V2 annotations are returned.

        negative_candidate:
            The image contains only explicitly approved
            organic / biodegradable classes.
            No positive labels are returned.

        ambiguous:
            The image contains target + non-target objects,
            or unsupported objects.
            No labels are returned because the whole image
            must be excluded from core training.

        empty:
            No source annotations.
            No assumption is made that it is a negative image.

    Invalid source annotations raise ValueError instead of
    being silently ignored.
    """
    parsed_annotations = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        parsed = parse_yolo_annotation(
            line,
            nc=nc,
        )

        parsed_annotations.append(
            {
                "line": line,
                "class_id": parsed["class_id"],
            }
        )

    source_classes = [
        annotation["class_id"]
        for annotation in parsed_annotations
    ]

    status = classify_image_classes(
        source_name,
        source_classes,
    )

    if status != "target":
        return {
            "status": status,
            "annotations": [],
        }

    final_annotations = []

    for annotation in parsed_annotations:
        converted = convert_yolo_annotation(
            source_name,
            annotation["line"],
            nc=nc,
        )

        if converted is None:
            raise ValueError(
                "Target image unexpectedly produced "
                "an unmapped annotation"
            )

        final_annotations.append(
            converted
        )

    return {
        "status": "target",
        "annotations": final_annotations,
    }