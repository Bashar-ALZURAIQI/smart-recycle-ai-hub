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


def test_pilot_training_configuration_uses_full_dataset_and_safe_settings():
    module = load_training_module()

    data_yaml = Path("data") / "processed" / "waste_v2_1" / "data.yaml"

    config = module.build_training_kwargs(
        mode="pilot",
        data_yaml=data_yaml,
        batch=8,
    )

    assert config["data"] == str(data_yaml)
    assert config["epochs"] == 1
    assert config["batch"] == 8
    assert config["imgsz"] == 640
    assert config["workers"] == 0
    assert config["cache"] is False
    assert config["seed"] == 26
    assert config["deterministic"] is True
    assert config["amp"] is True
    assert config["optimizer"] == "auto"
    assert config["device"] == 0
    assert config["name"] == "waste_v2_1_pilot"

    assert "fraction" not in config


def test_full_training_configuration_uses_eleven_epochs_and_early_stopping():
    module = load_training_module()

    data_yaml = Path("data") / "processed" / "waste_v2_1" / "data.yaml"

    config = module.build_training_kwargs(
        mode="full",
        data_yaml=data_yaml,
        batch=8,
    )

    assert config["data"] == str(data_yaml)
    assert config["epochs"] == 11
    assert config["patience"] == 3
    assert config["batch"] == 8
    assert config["imgsz"] == 640
    assert config["workers"] == 0
    assert config["cache"] is False
    assert config["seed"] == 26
    assert config["deterministic"] is True
    assert config["amp"] is True
    assert config["optimizer"] == "auto"
    assert config["device"] == 0
    assert config["name"] == "waste_v2_1_yolo26n"

    assert "fraction" not in config


def test_v2_1_training_uses_original_v2_best_checkpoint():
    module = load_training_module()

    weights = module.get_base_weights(PROJECT_ROOT)

    assert weights == (
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "waste_v2_yolo26n"
        / "weights"
        / "best.pt"
    )

    assert "waste_v2_1_pilot" not in str(weights)