import tempfile
import unittest
from pathlib import Path

from scripts.build_waste_v2 import resolve_dataset_splits


class TestRoboflowPathResolution(unittest.TestCase):

    def test_parent_relative_paths_fall_back_to_dataset_root(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            dataset_root = (
                Path(temp_directory)
                / "dataset"
            )

            train_images = (
                dataset_root
                / "train"
                / "images"
            )

            valid_images = (
                dataset_root
                / "valid"
                / "images"
            )

            test_images = (
                dataset_root
                / "test"
                / "images"
            )

            train_images.mkdir(
                parents=True
            )

            valid_images.mkdir(
                parents=True
            )

            test_images.mkdir(
                parents=True
            )

            (
                dataset_root
                / "data.yaml"
            ).write_text(
                (
                    "train: ../train/images\n"
                    "val: ../valid/images\n"
                    "test: ../test/images\n"
                ),
                encoding="utf-8",
            )

            resolved = dict(
                resolve_dataset_splits(
                    dataset_root
                )
            )

            self.assertEqual(
                resolved["train"],
                train_images.resolve(),
            )

            self.assertEqual(
                resolved["val"],
                valid_images.resolve(),
            )

            self.assertEqual(
                resolved["test"],
                test_images.resolve(),
            )


if __name__ == "__main__":
    unittest.main()