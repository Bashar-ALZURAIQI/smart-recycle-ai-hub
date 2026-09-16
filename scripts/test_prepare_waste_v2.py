import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


try:
    import prepare_waste_v2 as waste_v2
except ModuleNotFoundError:
    waste_v2 = None


class TestPrepareWasteV2(unittest.TestCase):

    def require_api(self, name):
        self.assertIsNotNone(
            waste_v2,
            "prepare_waste_v2.py does not exist yet",
        )

        self.assertTrue(
            hasattr(waste_v2, name),
            f"Expected {name} to exist",
        )

        return getattr(waste_v2, name)

    def test_source_class_mapping(self):
        map_source_class = self.require_api(
            "map_source_class"
        )

        cases = [
            ("waste_v1", 0, 0),
            ("waste_v1", 1, 1),
            ("waste_v1", 2, 2),
            ("waste_v1", 3, 3),

            ("garbage_classification", 0, None),
            ("garbage_classification", 1, 3),
            ("garbage_classification", 2, 2),
            ("garbage_classification", 3, 1),
            ("garbage_classification", 4, 3),
            ("garbage_classification", 5, 0),

            ("recyclable_waste_detection", 0, 3),
            ("recyclable_waste_detection", 1, 2),
            ("recyclable_waste_detection", 2, 1),
            ("recyclable_waste_detection", 3, 3),
            ("recyclable_waste_detection", 4, 0),

            ("conveyor_waste_belt", 0, 2),
            ("conveyor_waste_belt", 1, 1),
            ("conveyor_waste_belt", 2, None),
            ("conveyor_waste_belt", 3, 3),
            ("conveyor_waste_belt", 4, 0),

            ("taco_optional", 5, 0),
            ("taco_optional", 8, 1),
            ("taco_optional", 6, 2),
            ("taco_optional", 33, 3),

            ("taco_optional", 25, None),
            ("taco_optional", 58, None),
            ("taco_optional", 59, None),
        ]

        for source_name, source_class, expected in cases:
            with self.subTest(
                source=source_name,
                source_class=source_class,
            ):
                result = map_source_class(
                    source_name,
                    source_class,
                )

                self.assertEqual(
                    result,
                    expected,
                )

    def test_polygon_to_yolo_bbox(self):
        polygon_to_yolo_bbox = self.require_api(
            "polygon_to_yolo_bbox"
        )

        points = [
            (0.10, 0.20),
            (0.30, 0.20),
            (0.30, 0.40),
            (0.10, 0.40),
        ]

        bbox = polygon_to_yolo_bbox(points)

        x_center, y_center, width, height = bbox

        self.assertAlmostEqual(x_center, 0.20)
        self.assertAlmostEqual(y_center, 0.30)
        self.assertAlmostEqual(width, 0.20)
        self.assertAlmostEqual(height, 0.20)

    def test_convert_yolo_annotation_to_final_schema(self):
        convert_yolo_annotation = self.require_api(
            "convert_yolo_annotation"
        )

        cases = [
            {
                "source": "garbage_classification",
                "nc": 6,
                "line": "3 0.5 0.4 0.2 0.3",
                "expected": (
                    1,
                    0.5,
                    0.4,
                    0.2,
                    0.3,
                ),
            },
            {
                "source": "recyclable_waste_detection",
                "nc": 5,
                "line": (
                    "4 "
                    "0.10 0.20 "
                    "0.30 0.20 "
                    "0.30 0.40 "
                    "0.10 0.40"
                ),
                "expected": (
                    0,
                    0.20,
                    0.30,
                    0.20,
                    0.20,
                ),
            },
            {
                "source": "conveyor_waste_belt",
                "nc": 5,
                "line": "2 0.5 0.5 0.2 0.2",
                "expected": None,
            },
        ]

        for case in cases:
            result = convert_yolo_annotation(
                case["source"],
                case["line"],
                nc=case["nc"],
            )

            if case["expected"] is None:
                self.assertIsNone(result)
                continue

            self.assertEqual(
                result[0],
                case["expected"][0],
            )

            for actual, expected in zip(
                result[1:],
                case["expected"][1:],
            ):
                self.assertAlmostEqual(
                    actual,
                    expected,
                )

    def test_classify_image_classes(self):
        classify_image_classes = self.require_api(
            "classify_image_classes"
        )

        cases = [
            (
                "garbage_classification",
                [1, 4, 5],
                "target",
            ),
            (
                "recyclable_waste_detection",
                [0, 2, 4],
                "target",
            ),
            (
                "garbage_classification",
                [0],
                "negative_candidate",
            ),
            (
                "conveyor_waste_belt",
                [2],
                "negative_candidate",
            ),
            (
                "taco_optional",
                [25],
                "negative_candidate",
            ),
            (
                "garbage_classification",
                [0, 5],
                "ambiguous",
            ),
            (
                "conveyor_waste_belt",
                [2, 4],
                "ambiguous",
            ),
            (
                "taco_optional",
                [5, 25],
                "ambiguous",
            ),
            (
                "taco_optional",
                [59],
                "ambiguous",
            ),
            (
                "taco_optional",
                [58],
                "ambiguous",
            ),
            (
                "garbage_classification",
                [],
                "empty",
            ),
        ]

        for source_name, source_classes, expected in cases:
            with self.subTest(
                source=source_name,
                classes=source_classes,
            ):
                result = classify_image_classes(
                    source_name,
                    source_classes,
                )

                self.assertEqual(
                    result,
                    expected,
                )

    def test_prepare_yolo_image_annotations_for_target(self):
        prepare_yolo_image_annotations = self.require_api(
            "prepare_yolo_image_annotations"
        )

        lines = [
            "5 0.50 0.50 0.20 0.20",
            "3 0.25 0.30 0.10 0.12",
        ]

        result = prepare_yolo_image_annotations(
            "garbage_classification",
            lines,
            nc=6,
        )

        self.assertEqual(
            result["status"],
            "target",
        )

        self.assertEqual(
            len(result["annotations"]),
            2,
        )

        self.assertEqual(
            result["annotations"][0][0],
            0,
        )

        self.assertEqual(
            result["annotations"][1][0],
            1,
        )

    def test_prepare_yolo_image_annotations_handles_negative_and_ambiguous(self):
        prepare_yolo_image_annotations = self.require_api(
            "prepare_yolo_image_annotations"
        )

        negative = prepare_yolo_image_annotations(
            "garbage_classification",
            [
                "0 0.50 0.50 0.20 0.20",
            ],
            nc=6,
        )

        self.assertEqual(
            negative["status"],
            "negative_candidate",
        )

        self.assertEqual(
            negative["annotations"],
            [],
        )

        ambiguous = prepare_yolo_image_annotations(
            "garbage_classification",
            [
                "5 0.50 0.50 0.20 0.20",
                "0 0.30 0.30 0.10 0.10",
            ],
            nc=6,
        )

        self.assertEqual(
            ambiguous["status"],
            "ambiguous",
        )

        self.assertEqual(
            ambiguous["annotations"],
            [],
        )


if __name__ == "__main__":
    unittest.main()