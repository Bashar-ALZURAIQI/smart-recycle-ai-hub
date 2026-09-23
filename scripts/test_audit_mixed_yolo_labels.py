import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


import audit_waste_v2_sources as audit


class TestMixedYoloAudit(unittest.TestCase):

    def test_parse_yolo_annotation_recognizes_segmentation(self):
        self.assertTrue(
            hasattr(audit, "parse_yolo_annotation"),
            "Expected parse_yolo_annotation() to exist",
        )

        result = audit.parse_yolo_annotation(
            "4 0.10 0.20 0.30 0.20 0.30 0.40 0.10 0.40",
            nc=5,
        )

        self.assertEqual(
            result["kind"],
            "segmentation",
        )

        self.assertEqual(
            result["class_id"],
            4,
        )

        self.assertEqual(
            len(result["points"]),
            4,
        )

    def test_count_yolo_classes_separates_bbox_segmentation_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "mixed.txt"

            label_path.write_text(
                "\n".join(
                    [
                        # Standard detection bbox.
                        "0 0.5 0.5 0.2 0.2",

                        # Valid segmentation polygon.
                        "4 0.10 0.20 0.30 0.20 0.30 0.40 0.10 0.40",

                        # Degenerate detection bbox: width = 0.
                        "1 0.5 0.5 0 0.2",
                    ]
                ),
                encoding="utf-8",
            )

            report = audit.count_yolo_classes(
                [label_path],
                nc=5,
            )

            self.assertEqual(
                report["class_counts"],
                {
                    0: 1,
                    4: 1,
                },
            )

            self.assertEqual(
                report["valid_annotations"],
                2,
            )

            self.assertEqual(
                report["bbox_annotations"],
                1,
            )

            self.assertEqual(
                report["segmentation_annotations"],
                1,
            )

            self.assertEqual(
                report["invalid_lines"],
                1,
            )


if __name__ == "__main__":
    unittest.main()