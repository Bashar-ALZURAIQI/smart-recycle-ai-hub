import argparse

from pathlib import Path


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


def require_cuda(torch_module) -> dict:
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Waste V2.1 training; "
            "CPU fallback is not allowed."
        )

    device_index = 0
    gpu_name = torch_module.cuda.get_device_name(device_index)
    gpu_properties = torch_module.cuda.get_device_properties(device_index)
    vram_gb = gpu_properties.total_memory / (1024**3)

    return {
        "device_index": device_index,
        "name": gpu_name,
        "vram_gb": vram_gb,
    }


def get_base_weights(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "runs"
        / "detect"
        / "waste_v2_yolo26n"
        / "weights"
        / "best.pt"
    )


def build_training_kwargs(
    mode: str,
    data_yaml: Path,
    batch: int,
) -> dict:
    if mode not in {"pilot", "full"}:
        raise ValueError(
            f"unsupported training mode: {mode}"
        )

    config = {
        "data": str(data_yaml),
        "batch": batch,
        "imgsz": 640,
        "workers": 0,
        "cache": False,
        "seed": 26,
        "deterministic": True,
        "amp": True,
        "optimizer": "auto",
        "device": 0,
    }

    if mode == "pilot":
        config["epochs"] = 1
        config["name"] = "waste_v2_1_pilot"
    else:
        config["epochs"] = 11
        config["patience"] = 3
        config["name"] = "waste_v2_1_yolo26n"

    return config


def summarize_split(dataset_root: Path, split: str) -> dict:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_files = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    label_files = sorted(
        path
        for path in label_dir.glob("*.txt")
        if path.is_file()
    )

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        raise ValueError(
            f"{split} image/label mismatch: "
            f"missing_labels={missing_labels}, "
            f"missing_images={missing_images}"
        )

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES.values()
    }

    empty_labels = 0
    annotations = 0

    for label_path in label_files:
        text = label_path.read_text(encoding="utf-8").strip()

        if not text:
            empty_labels += 1
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"{split} five-token YOLO label required: "
                    f"{label_path.name}:{line_number}"
                )

            class_id = int(parts[0])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"{split} invalid class id {class_id}: "
                    f"{label_path.name}:{line_number}"
                )

            x, y, width, height = map(float, parts[1:])

            valid_geometry = (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            )

            if not valid_geometry:
                raise ValueError(
                    f"{split} invalid bbox geometry: "
                    f"{label_path.name}:{line_number}"
                )

            class_counts[CLASS_NAMES[class_id]] += 1
            annotations += 1

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "empty_labels": empty_labels,
        "annotations": annotations,
        "classes": class_counts,
    }


def preflight_dataset(data_yaml: Path) -> dict:
    data_yaml = Path(data_yaml)

    if not data_yaml.is_file():
        raise ValueError(
            f"data.yaml does not exist: {data_yaml}"
        )

    dataset_root = data_yaml.parent

    required_directories = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
    ]

    for directory in required_directories:
        if not directory.is_dir():
            raise ValueError(
                f"required dataset directory does not exist: {directory}"
            )

    return {
        "train": summarize_split(dataset_root, "train"),
        "valid": summarize_split(dataset_root, "valid"),
    }


def run_training(
    mode: str,
    data_yaml: Path,
    batch: int,
    project_root: Path,
    torch_module,
    yolo_factory,
):
    preflight_dataset(data_yaml)

    require_cuda(torch_module)

    weights = get_base_weights(project_root)

    model = yolo_factory(weights)

    training_kwargs = build_training_kwargs(
        mode=mode,
        data_yaml=data_yaml,
        batch=batch,
    )

    return model.train(**training_kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train Waste V2.1 with YOLO26n."
    )

    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        required=True,
        help="Run the one-epoch pilot or the full training.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        choices=(4, 8),
        help="Training batch size. Default: 8.",
    )

    return parser.parse_args(argv)


def main(
    argv=None,
    *,
    torch_module=None,
    yolo_factory=None,
    project_root=None,
):
    args = parse_args(argv)

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root)

    if torch_module is None:
        import torch

        torch_module = torch

    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO

    data_yaml = (
        project_root
        / "data"
        / "processed"
        / "waste_v2_1"
        / "data.yaml"
    )

    gpu_info = require_cuda(torch_module)

    print(
        "GPU: "
        f"{gpu_info['name']} | "
        f"VRAM: {gpu_info['vram_gb']:.2f} GB | "
        f"device={gpu_info['device_index']}"
    )

    print(
        "Training mode: "
        f"{args.mode} | "
        f"batch={args.batch}"
    )

    print(
        "Base checkpoint: "
        f"{get_base_weights(project_root)}"
    )

    return run_training(
        mode=args.mode,
        data_yaml=data_yaml,
        batch=args.batch,
        project_root=project_root,
        torch_module=torch_module,
        yolo_factory=yolo_factory,
    )


if __name__ == "__main__":
    main()