import pytest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_waste_v2_1.py"


def load_training_module():
    spec = spec_from_file_location("train_waste_v2_1", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_dataset_rejects_missing_data_yaml(tmp_path):
    module = load_training_module()

    missing_data_yaml = tmp_path / "missing_dataset" / "data.yaml"

    with pytest.raises(ValueError, match="data.yaml does not exist"):
        module.preflight_dataset(missing_data_yaml)


@pytest.mark.parametrize(
    "missing_relative_path",
    [
        Path("train") / "images",
        Path("train") / "labels",
        Path("valid") / "images",
        Path("valid") / "labels",
    ],
)
def test_preflight_dataset_rejects_missing_required_directory(
    tmp_path,
    missing_relative_path,
):
    dataset_root = tmp_path / "waste_v2_1"

    required_directories = [
        Path("train") / "images",
        Path("train") / "labels",
        Path("valid") / "images",
        Path("valid") / "labels",
    ]

    for relative_path in required_directories:
        if relative_path != missing_relative_path:
            (dataset_root / relative_path).mkdir(
                parents=True,
                exist_ok=True,
            )

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

    module = load_training_module()

    with pytest.raises(ValueError, match="required dataset directory"):
        module.preflight_dataset(data_yaml)