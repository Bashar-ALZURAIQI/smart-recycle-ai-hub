import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


import prepare_waste_v2 as waste_v2


class TestWasteV2LeakageAndFamilies(unittest.TestCase):

    def require_api(self, name):
        self.assertTrue(
            hasattr(waste_v2, name),
            f"Expected {name} to exist",
        )

        return getattr(waste_v2, name)

    def make_pattern_image(
        self,
        path,
        reverse=False,
    ):
        image = Image.new(
            "RGB",
            (32, 32),
        )

        pixels = image.load()

        for y in range(32):
            for x in range(32):
                value = (
                    255 - (x * 8 % 256)
                    if reverse
                    else (x * 8 % 256)
                )

                pixels[x, y] = (
                    value,
                    (y * 8) % 256,
                    (x + y) * 4 % 256,
                )

        image.save(path)

    def test_roboflow_augmented_siblings_share_family(self):
        roboflow_family_key = self.require_api(
            "roboflow_family_key"
        )

        first = (
            "plastic_001_jpg."
            "rf.11111111111111111111111111111111.jpg"
        )

        second = (
            "plastic_001_jpg."
            "rf.22222222222222222222222222222222.jpg"
        )

        self.assertEqual(
            roboflow_family_key(first),
            roboflow_family_key(second),
        )

        self.assertEqual(
            roboflow_family_key(first),
            "plastic_001_jpg",
        )

        self.assertEqual(
            roboflow_family_key("ordinary_image.jpg"),
            "ordinary_image",
        )

    def test_family_id_keeps_different_sources_separate(self):
        image_family_id = self.require_api(
            "image_family_id"
        )

        filename = (
            "plastic_001_jpg."
            "rf.11111111111111111111111111111111.jpg"
        )

        first = image_family_id(
            "recyclable_waste_detection",
            filename,
        )

        second = image_family_id(
            "recyclable_waste_detection",
            (
                "plastic_001_jpg."
                "rf.22222222222222222222222222222222.jpg"
            ),
        )

        other_source = image_family_id(
            "conveyor_waste_belt",
            filename,
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertNotEqual(
            first,
            other_source,
        )

    def test_image_fingerprint_detects_exact_and_perceptual_match(self):
        image_fingerprint = self.require_api(
            "image_fingerprint"
        )

        benchmark_match_kind = self.require_api(
            "benchmark_match_kind"
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            benchmark = root / "benchmark.png"
            exact_copy = root / "exact_copy.png"
            reencoded = root / "reencoded.bmp"
            different = root / "different.png"

            self.make_pattern_image(
                benchmark
            )

            exact_copy.write_bytes(
                benchmark.read_bytes()
            )

            with Image.open(benchmark) as image:
                image.save(
                    reencoded,
                    format="BMP",
                )

            self.make_pattern_image(
                different,
                reverse=True,
            )

            benchmark_fp = image_fingerprint(
                benchmark
            )

            exact_fp = image_fingerprint(
                exact_copy
            )

            reencoded_fp = image_fingerprint(
                reencoded
            )

            different_fp = image_fingerprint(
                different
            )

            self.assertEqual(
                benchmark_match_kind(
                    exact_fp,
                    [benchmark_fp],
                ),
                "exact",
            )

            self.assertEqual(
                benchmark_match_kind(
                    reencoded_fp,
                    [benchmark_fp],
                ),
                "perceptual_exact",
            )

            self.assertIsNone(
                benchmark_match_kind(
                    different_fp,
                    [benchmark_fp],
                )
            )

    def test_group_records_by_family_keeps_siblings_together(self):
        group_records_by_family = self.require_api(
            "group_records_by_family"
        )

        records = [
            {
                "source": "recyclable_waste_detection",
                "path": (
                    "train/images/"
                    "bottle_jpg."
                    "rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"
                ),
            },
            {
                "source": "recyclable_waste_detection",
                "path": (
                    "valid/images/"
                    "bottle_jpg."
                    "rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"
                ),
            },
            {
                "source": "recyclable_waste_detection",
                "path": (
                    "train/images/"
                    "metal_001_jpg."
                    "rf.cccccccccccccccccccccccccccccccc.jpg"
                ),
            },
            {
                "source": "conveyor_waste_belt",
                "path": (
                    "train/images/"
                    "bottle_jpg."
                    "rf.dddddddddddddddddddddddddddddddd.jpg"
                ),
            },
        ]

        groups = group_records_by_family(
            records
        )

        group_sizes = sorted(
            len(group)
            for group in groups.values()
        )

        self.assertEqual(
            group_sizes,
            [1, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()