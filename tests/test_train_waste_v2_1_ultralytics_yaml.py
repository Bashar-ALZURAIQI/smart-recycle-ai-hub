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


def test_prepare_ultralytics_data_yaml_uses_absolute_dataset_root(
    tmp_path,
):
    module = load_training_module()

    dataset_root = tmp_path / "waste_v2_1"
    dataset_root.mkdir(parents=True)

    source_yaml = dataset_root / "data.yaml"

    source_yaml.write_text(
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

    runtime_yaml = module.prepare_ultralytics_data_yaml(
        source_yaml
    )

    assert runtime_yaml == (
        dataset_root / "data.ultralytics.yaml"
    )

    runtime_text = runtime_yaml.read_text(
        encoding="utf-8"
    )

    expected_root = dataset_root.resolve().as_posix()

    assert f"path: {expected_root}" in runtime_text
    assert "train: train/images" in runtime_text
    assert "val: valid/images" in runtime_text

    original_text = source_yaml.read_text(
        encoding="utf-8"
    )

    assert "path: ." in original_text
    assert expected_root not in original_text