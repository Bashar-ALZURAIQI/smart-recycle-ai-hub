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


def write_dataset(dataset_root: Path) -> Path:
    for split in ("train", "valid"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        (image_dir / f"{split}_sample.jpg").write_bytes(b"fake-image")
        (label_dir / f"{split}_sample.txt").write_text(
            "0 0.50 0.50 0.20 0.20\n",
            encoding="utf-8",
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

    return data_yaml


class FakeDeviceProperties:
    total_memory = 6 * 1024**3


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_name(device_index):
        assert device_index == 0
        return "Test RTX GPU"

    @staticmethod
    def get_device_properties(device_index):
        assert device_index == 0
        return FakeDeviceProperties()


class FakeTorch:
    cuda = FakeCuda()


class FakeModel:
    def __init__(self):
        self.train_calls = []

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        return "fake-training-result"


@pytest.mark.parametrize(
    "mode",
    [
        "pilot",
        "full",
    ],
)
def test_run_training_uses_original_v2_checkpoint_for_every_mode(
    tmp_path,
    mode,
):
    module = load_training_module()

    project_root = tmp_path / "project"
    dataset_root = project_root / "data" / "processed" / "waste_v2_1"
    data_yaml = write_dataset(dataset_root)

    expected_weights = (
        project_root
        / "runs"
        / "detect"
        / "waste_v2_yolo26n"
        / "weights"
        / "best.pt"
    )

    expected_weights.parent.mkdir(parents=True, exist_ok=True)
    expected_weights.write_bytes(b"fake-checkpoint")

    created_models = []

    def fake_yolo_factory(weights):
        assert Path(weights) == expected_weights

        model = FakeModel()
        created_models.append(model)
        return model

    result = module.run_training(
        mode=mode,
        data_yaml=data_yaml,
        batch=8,
        project_root=project_root,
        torch_module=FakeTorch,
        yolo_factory=fake_yolo_factory,
    )

    assert result == "fake-training-result"
    assert len(created_models) == 1
    assert len(created_models[0].train_calls) == 1

    train_kwargs = created_models[0].train_calls[0]

    assert train_kwargs["data"] == str(data_yaml)
    assert train_kwargs["batch"] == 8
    assert train_kwargs["device"] == 0

    if mode == "pilot":
        assert train_kwargs["epochs"] == 1
        assert train_kwargs["name"] == "waste_v2_1_pilot"
    else:
        assert train_kwargs["epochs"] == 11
        assert train_kwargs["patience"] == 3
        assert train_kwargs["name"] == "waste_v2_1_yolo26n"