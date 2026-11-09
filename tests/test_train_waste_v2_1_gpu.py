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


class FakeCudaUnavailable:
    @staticmethod
    def is_available():
        return False


class FakeTorchWithoutCuda:
    cuda = FakeCudaUnavailable()
    __version__ = "test"


class FakeDeviceProperties:
    total_memory = 6 * 1024**3


class FakeCudaAvailable:
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


class FakeTorchWithCuda:
    cuda = FakeCudaAvailable()
    __version__ = "test"


def test_require_cuda_rejects_cpu_only_environment():
    module = load_training_module()

    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        module.require_cuda(FakeTorchWithoutCuda)


def test_require_cuda_reports_gpu_name_and_vram():
    module = load_training_module()

    gpu_info = module.require_cuda(FakeTorchWithCuda)

    assert gpu_info == {
        "device_index": 0,
        "name": "Test RTX GPU",
        "vram_gb": 6.0,
    }