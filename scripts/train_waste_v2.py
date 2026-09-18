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

    image_files = [
        path
        for path
        in images_directory.iterdir()
        if path.is_file()
    ]

    label_files = list(
        labels_directory.glob(
            "*.txt"
        )
    )

    if (
        len(image_files)
        != len(label_files)
    ):
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
                        f"{label_path}: "
                        f"{line}"
                    )
                )

            class_id = int(
                parts[0]
            )

            if (
                class_id
                not in CLASS_NAMES
            ):

                raise RuntimeError(
                    (
                        "Unexpected class ID "
                        f"{class_id} in "
                        f"{label_path}"
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
            CLASS_NAMES[class_id]:
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
                "Training was stopped to avoid "
                "running the job on CPU."
            )
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    total_memory = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / 1024**3
    )

    print(
        "GPU memory:",
        f"{total_memory:.2f} GB",
    )


def clear_cuda_memory() -> None:

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


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
        "Batch size: 8"
    )

    print(
        "Workers: 0"
    )

    print(
        (
            "Workers are disabled to reduce "
            "Windows/OpenCV RAM pressure."
        )
    )

    model = YOLO(
        str(
            weights_path
        )
    )

    common_arguments = {
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
        "project": str(
            RUNS_DIRECTORY
        ),
        "plots": True,
        "verbose": True,
    }

    if mode == "sanity":

        print(
            "\n=== SANITY TRAINING ==="
        )

        print(
            (
                "Running 1 epoch on "
                "5% of the training dataset."
            )
        )

        print(
            (
                "Validation still uses the "
                "full validation split."
            )
        )

        model.train(
            epochs=1,
            fraction=0.05,
            name="waste_v2_sanity",
            exist_ok=True,
            **common_arguments,
        )

        sanity_best = (
            RUNS_DIRECTORY
            / "waste_v2_sanity"
            / "weights"
            / "best.pt"
        )

        print(
            "\n=== SANITY COMPLETE ==="
        )

        print(
            "Sanity best.pt:"
        )

        print(
            sanity_best
        )

        print(
            (
                "\nSanity passed. "
                "Use the ORIGINAL V1 best.pt "
                "for the full V2 training."
            )
        )

    elif mode == "full":

        print(
            "\n=== FULL V2 TRAINING ==="
        )

        print(
            (
                "Fine-tuning YOLO26n "
                "for up to 80 epochs."
            )
        )

        model.train(
            epochs=80,
            patience=20,
            save=True,
            save_period=10,
            name="waste_v2_yolo26n",
            exist_ok=True,
            **common_arguments,
        )

        expected_best = (
            RUNS_DIRECTORY
            / "waste_v2_yolo26n"
            / "weights"
            / "best.pt"
        )

        expected_last = (
            RUNS_DIRECTORY
            / "waste_v2_yolo26n"
            / "weights"
            / "last.pt"
        )

        print(
            "\n=== FULL TRAINING COMPLETE ==="
        )

        print(
            "Expected best.pt:"
        )

        print(
            expected_best
        )

        print(
            "\nExpected last.pt:"
        )

        print(
            expected_last
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
            "Train YOLO26n on "
            "Smart Recycle Waste V2."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "Path to the V1 YOLO26n "
            "best.pt checkpoint."
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