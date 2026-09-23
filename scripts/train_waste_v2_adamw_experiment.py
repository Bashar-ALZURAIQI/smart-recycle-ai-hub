from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "waste_v2"
)

DATA_YAML = (
    DATASET_ROOT
    / "data.yaml"
)

RUNS_DIRECTORY = (
    PROJECT_ROOT
    / "runs"
    / "detect"
)

CLASS_NAMES = {
    0: "plastic",
    1: "metal",
    2: "glass",
    3: "paper_cardboard",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def inspect_split(
    split_name: str,
) -> dict:

    images_directory = (
        DATASET_ROOT
        / split_name
        / "images"
    )

    labels_directory = (
        DATASET_ROOT
        / split_name
        / "labels"
    )

    if not images_directory.exists():
        raise FileNotFoundError(
            f"Missing images directory: {images_directory}"
        )

    if not labels_directory.exists():
        raise FileNotFoundError(
            f"Missing labels directory: {labels_directory}"
        )

    image_files = [
        path
        for path in images_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    label_files = list(
        labels_directory.rglob("*.txt")
    )

    if len(image_files) != len(label_files):
        raise RuntimeError(
            (
                f"{split_name}: "
                "image/label count mismatch: "
                f"{len(image_files)} images, "
                f"{len(label_files)} labels"
            )
        )

    class_counts = Counter()

    empty_labels = 0
    annotation_count = 0

    for label_path in label_files:

        lines = [
            line.strip()
            for line
            in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                raise RuntimeError(
                    (
                        "Invalid YOLO label line in "
                        f"{label_path}: {line}"
                    )
                )

            try:
                class_id = int(
                    parts[0]
                )

                coordinates = [
                    float(value)
                    for value
                    in parts[1:]
                ]

            except ValueError as error:
                raise RuntimeError(
                    (
                        "Invalid YOLO label in "
                        f"{label_path}: {line}"
                    )
                ) from error

            if class_id not in CLASS_NAMES:
                raise RuntimeError(
                    (
                        "Unexpected class ID "
                        f"{class_id} in "
                        f"{label_path}"
                    )
                )

            x_center, y_center, width, height = (
                coordinates
            )

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise RuntimeError(
                    (
                        "Invalid bounding box in "
                        f"{label_path}: {line}"
                    )
                )

            class_counts[
                class_id
            ] += 1

            annotation_count += 1

    return {
        "images": len(
            image_files
        ),
        "labels": len(
            label_files
        ),
        "empty_labels": (
            empty_labels
        ),
        "annotations": (
            annotation_count
        ),
        "classes": {
            CLASS_NAMES[
                class_id
            ]:
            class_counts[
                class_id
            ]
            for class_id
            in CLASS_NAMES
        },
    }


def run_preflight() -> None:

    print(
        "\n=== WASTE V2 PREFLIGHT ==="
    )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            (
                "Dataset YAML not found: "
                f"{DATA_YAML}"
            )
        )

    train_stats = inspect_split(
        "train"
    )

    valid_stats = inspect_split(
        "valid"
    )

    report = {
        "train": train_stats,
        "valid": valid_stats,
    }

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    for (
        split_name,
        stats,
    ) in (
        (
            "train",
            train_stats,
        ),
        (
            "valid",
            valid_stats,
        ),
    ):

        for (
            class_name,
            count,
        ) in stats[
            "classes"
        ].items():

            if count == 0:
                raise RuntimeError(
                    (
                        f"{split_name} has "
                        "zero annotations for "
                        f"{class_name}"
                    )
                )

    print(
        "\nPreflight: OK"
    )


def verify_gpu() -> None:

    print(
        "\n=== GPU CHECK ==="
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            (
                "CUDA is not available. "
                "Training stopped."
            )
        )

    gpu_name = (
        torch.cuda.get_device_name(
            0
        )
    )

    gpu_memory = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / 1024**3
    )

    print(
        "GPU:",
        gpu_name,
    )

    print(
        "GPU memory:",
        f"{gpu_memory:.2f} GB",
    )


def clear_cuda_memory() -> None:

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_training_configuration(
    weights_path: Path,
    mode: str,
) -> None:

    print(
        "\n=== TRAINING CONFIGURATION ==="
    )

    print(
        "Mode:",
        mode,
    )

    print(
        "Starting weights:",
        weights_path,
    )

    print(
        "Dataset:",
        DATA_YAML,
    )

    print(
        "Image size: 640"
    )

    print(
        "Batch size: 8"
    )

    print(
        "Workers: 0"
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        "Initial LR: 0.001"
    )

    print(
        "Final LR factor: 0.1"
    )

    print(
        "Warmup epochs: 1"
    )

    print(
        "Full epochs: 20"
    )

    print(
        "Early stopping patience: 5"
    )


def build_common_arguments() -> dict:

    return {
        "data": str(
            DATA_YAML
        ),
        "imgsz": 640,
        "device": 0,
        "batch": 8,
        "workers": 0,
        "cache": False,
        "seed": 26,
        "deterministic": True,
        "amp": True,

        # Stable fine-tuning configuration
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.1,

        # Short and conservative warmup
        "warmup_epochs": 1.0,
        "warmup_bias_lr": 0.001,

        "project": str(
            RUNS_DIRECTORY
        ),

        "plots": True,
        "verbose": True,
    }


def run_sanity(
    model: YOLO,
    common_arguments: dict,
) -> None:

    print(
        "\n=== ADAMW SANITY TRAINING ==="
    )

    print(
        (
            "Running one epoch using "
            "5% of the training dataset."
        )
    )

    model.train(
        epochs=1,
        fraction=0.05,
        patience=1,
        name="waste_v2_adamw_sanity",
        exist_ok=True,
        **common_arguments,
    )

    sanity_directory = (
        RUNS_DIRECTORY
        / "waste_v2_adamw_sanity"
    )

    print(
        "\n=== SANITY COMPLETE ==="
    )

    print(
        "Results directory:"
    )

    print(
        sanity_directory
    )


def run_full_training(
    model: YOLO,
    common_arguments: dict,
) -> None:

    print(
        "\n=== OFFICIAL V2 ADAMW FINE-TUNING ==="
    )

    print(
        (
            "Starting a NEW training run. "
            "The previous MuSGD run "
            "will remain untouched."
        )
    )

    print(
        "\nMaximum epochs: 20"
    )

    print(
        "Early stopping patience: 5"
    )

    model.train(
        epochs=20,
        patience=5,
        save=True,
        save_period=5,

        name=(
            "waste_v2_yolo26n_adamw_ft"
        ),

        exist_ok=True,

        **common_arguments,
    )

    run_directory = (
        RUNS_DIRECTORY
        / "waste_v2_yolo26n_adamw_ft"
    )

    best_weights = (
        run_directory
        / "weights"
        / "best.pt"
    )

    last_weights = (
        run_directory
        / "weights"
        / "last.pt"
    )

    print(
        "\n=== V2 ADAMW TRAINING COMPLETE ==="
    )

    print(
        "Run directory:"
    )

    print(
        run_directory
    )

    print(
        "\nBest weights:"
    )

    print(
        best_weights
    )

    print(
        "\nLast weights:"
    )

    print(
        last_weights
    )


def train(
    weights_path: Path,
    mode: str,
) -> None:

    weights_path = (
        weights_path.resolve()
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            (
                "Weights not found: "
                f"{weights_path}"
            )
        )

    run_preflight()

    verify_gpu()

    clear_cuda_memory()

    print_training_configuration(
        weights_path,
        mode,
    )

    model = YOLO(
        str(
            weights_path
        )
    )

    common_arguments = (
        build_common_arguments()
    )

    if mode == "sanity":

        run_sanity(
            model,
            common_arguments,
        )

    elif mode == "full":

        run_full_training(
            model,
            common_arguments,
        )

    else:

        raise ValueError(
            (
                "mode must be "
                "'sanity' or 'full'"
            )
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Stable YOLO26n Waste V2 "
            "fine-tuning pipeline."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "Path to the original "
            "V1 best.pt checkpoint."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "sanity",
            "full",
        ],
        required=True,
    )

    arguments = (
        parser.parse_args()
    )

    train(
        weights_path=(
            arguments.weights
        ),
        mode=(
            arguments.mode
        ),
    )


if __name__ == "__main__":
    main()