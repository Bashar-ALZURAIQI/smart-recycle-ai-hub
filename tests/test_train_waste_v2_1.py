import pytest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_waste_v2_1.py"


def load_training_module():
    assert TRAIN_SCRIPT.exists(), (
        "Waste V2.1 training script does not exist yet: "
        f"{TRAIN_SCRIPT}"
    )

    spec = spec_from_file_location("train_waste_v2_1", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_yolo_pair(
    dataset_root: Path,
    split: str,
    stem: str,
    labels: str,
) -> None:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    (image_dir / f"{stem}.jpg").write_bytes(b"fake-image")
    (label_dir / f"{stem}.txt").write_text(labels, encoding="utf-8")


def write_data_yaml(dataset_root: Path) -> Path:
    data_yaml = dataset_root / "data.yaml"

    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: train/images",
                "val: valid/images",
                "",
                "names:",
                "  0: plastic",
                "  1: metal",
                "  2: glass",
                "  3: paper_cardboard",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return data_yaml


def test_preflight_dataset_reports_train_valid_and_class_counts(tmp_path):
    dataset_root = tmp_path / "waste_v2_1"

    write_yolo_pair(
        dataset_root,
        "train",
        "train_positive",
        "0 0.50 0.50 0.20 0.20\n"
        "3 0.40 0.40 0.10 0.10\n",
    )

    write_yolo_pair(
        dataset_root,
        "train",
        "train_negative",
        "",
    )

    write_yolo_pair(
        dataset_root,
        "valid",
        "valid_positive",
        "1 0.50 0.50 0.20 0.20\n"
        "2 0.40 0.40 0.10 0.10\n",
    )

    data_yaml = write_data_yaml(dataset_root)

    module = load_training_module()

    report = module.preflight_dataset(data_yaml)

    assert report == {
        "train": {
            "images": 2,
            "labels": 2,
            "empty_labels": 1,
            "annotations": 2,
            "classes": {
                "plastic": 1,
                "metal": 0,
                "glass": 0,
                "paper_cardboard": 1,
            },
        },
        "valid": {
            "images": 1,
            "labels": 1,
            "empty_labels": 0,
            "annotations": 2,
            "classes": {
                "plastic": 0,
                "metal": 1,
                "glass": 1,
                "paper_cardboard": 0,
            },
        },
    }


def test_preflight_dataset_rejects_image_label_stem_mismatch(tmp_path):
    dataset_root = tmp_path / "waste_v2_1"

    train_image_dir = dataset_root / "train" / "images"
    train_label_dir = dataset_root / "train" / "labels"

    train_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)

    (train_image_dir / "image_a.jpg").write_bytes(b"fake-image")
    (train_label_dir / "different_name.txt").write_text(
        "0 0.50 0.50 0.20 0.20\n",
        encoding="utf-8",
    )

    write_yolo_pair(
        dataset_root,
        "valid",
        "valid_ok",
        "1 0.50 0.50 0.20 0.20\n",
    )

    data_yaml = write_data_yaml(dataset_root)

    module = load_training_module()

    with pytest.raises(ValueError, match="image/label mismatch"):
        module.preflight_dataset(data_yaml)


def test_preflight_dataset_rejects_non_five_token_label(tmp_path):
    dataset_root = tmp_path / "waste_v2_1"

    write_yolo_pair(
        dataset_root,
        "train",
        "bad_label",
        "0 0.50 0.50 0.20\n",
    )

    write_yolo_pair(
        dataset_root,
        "valid",
        "valid_ok",
        "1 0.50 0.50 0.20 0.20\n",
    )

    data_yaml = write_data_yaml(dataset_root)

    module = load_training_module()

    with pytest.raises(ValueError, match="five-token"):
        module.preflight_dataset(data_yaml)


def test_preflight_dataset_rejects_invalid_class_id(tmp_path):
    dataset_root = tmp_path / "waste_v2_1"

    write_yolo_pair(
        dataset_root,
        "train",
        "bad_class",
        "4 0.50 0.50 0.20 0.20\n",
    )

    write_yolo_pair(
        dataset_root,
        "valid",
        "valid_ok",
        "1 0.50 0.50 0.20 0.20\n",
    )

    data_yaml = write_data_yaml(dataset_root)

    module = load_training_module()

    with pytest.raises(ValueError, match="invalid class id"):
        module.preflight_dataset(data_yaml)


@pytest.mark.parametrize(
    "bad_label",
    [
        "0 -0.01 0.50 0.20 0.20\n",
        "0 1.01 0.50 0.20 0.20\n",
        "0 0.50 -0.01 0.20 0.20\n",
        "0 0.50 1.01 0.20 0.20\n",
        "0 0.50 0.50 0.00 0.20\n",
        "0 0.50 0.50 1.01 0.20\n",
        "0 0.50 0.50 0.20 0.00\n",
        "0 0.50 0.50 0.20 1.01\n",
    ],
)
def test_preflight_dataset_rejects_invalid_bbox_geometry(
    tmp_path,
    bad_label,
):
    dataset_root = tmp_path / "waste_v2_1"

    write_yolo_pair(
        dataset_root,
        "train",
        "bad_bbox",
        bad_label,
    )

    write_yolo_pair(
        dataset_root,
        "valid",
        "valid_ok",
        "1 0.50 0.50 0.20 0.20\n",
    )

    data_yaml = write_data_yaml(dataset_root)

    module = load_training_module()

    with pytest.raises(ValueError, match="invalid bbox geometry"):
        module.preflight_dataset(data_yaml)