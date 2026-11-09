from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_waste_v2_1.py"


def load_training_module():
    spec = spec_from_file_location(
        "train_waste_v2_1",
        TRAIN_SCRIPT,
    )
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

        (image_dir / f"{split}_sample.jpg").write_bytes(
            b"fake-image"
        )

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


def test_run_training_uses_ultralytics_runtime_yaml(
    tmp_path,
):
    module = load_training_module()

    project_root = tmp_path / "project"

    dataset_root = (
        project_root
        / "data"
        / "processed"
        / "waste_v2_1"
    )

    source_yaml = write_dataset(dataset_root)

    created_models = []

    def fake_yolo_factory(weights):
        model = FakeModel()
        created_models.append(model)
        return model

    result = module.run_training(
        mode="pilot",
        data_yaml=source_yaml,
        batch=8,
        project_root=project_root,
        torch_module=FakeTorch,
        yolo_factory=fake_yolo_factory,
    )

    assert result == "fake-training-result"
    assert len(created_models) == 1

    train_kwargs = created_models[0].train_calls[0]

    runtime_yaml = (
        dataset_root / "data.ultralytics.yaml"
    )

    assert train_kwargs["data"] == str(runtime_yaml)
    assert runtime_yaml.is_file()

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