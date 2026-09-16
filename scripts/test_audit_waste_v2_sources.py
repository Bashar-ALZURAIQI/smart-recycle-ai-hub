import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


try:
    import audit_waste_v2_sources as audit
except ModuleNotFoundError:
    audit = None


class TestWasteV2SourceAudit(unittest.TestCase):

    def require_api(self, name):
        self.assertIsNotNone(
            audit,
            "audit_waste_v2_sources.py does not exist yet",
        )

        self.assertTrue(
            hasattr(audit, name),
            f"Expected function {name}() does not exist yet",
        )

        return getattr(audit, name)

    def test_find_images_recursively(self):
        find_images = self.require_api("find_images")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / "train").mkdir()
            (root / "valid").mkdir()

            (root / "train" / "plastic.jpg").write_bytes(b"fake")
            (root / "valid" / "metal.PNG").write_bytes(b"fake")
            (root / "valid" / "notes.txt").write_text(
                "not an image",
                encoding="utf-8",
            )

            images = find_images(root)

            names = sorted(path.name for path in images)

            self.assertEqual(
                names,
                ["metal.PNG", "plastic.jpg"],
            )

    def test_validate_yolo_line(self):
        validate_yolo_line = self.require_api("validate_yolo_line")

        result = validate_yolo_line(
            "2 0.5 0.4 0.2 0.3",
            nc=4,
        )

        self.assertEqual(
            result,
            (2, 0.5, 0.4, 0.2, 0.3),
        )

        invalid_lines = [
            "4 0.5 0.5 0.2 0.2",
            "1 0.5 0.5 0 0.2",
            "1 1.1 0.5 0.2 0.2",
            "1 0.5 0.5",
        ]

        for line in invalid_lines:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    validate_yolo_line(line, nc=4)

    def test_count_yolo_classes_reports_invalid_lines(self):
        count_yolo_classes = self.require_api("count_yolo_classes")

        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "sample.txt"

            label_path.write_text(
                "\n".join(
                    [
                        "0 0.5 0.5 0.2 0.2",
                        "3 0.4 0.4 0.1 0.1",
                        "0 0.3 0.3 0.1 0.1",
                        "9 0.5 0.5 0.2 0.2",
                        "bad line",
                    ]
                ),
                encoding="utf-8",
            )

            report = count_yolo_classes(
                [label_path],
                nc=4,
            )

            self.assertEqual(
                report["class_counts"],
                {0: 2, 3: 1},
            )

            self.assertEqual(
                report["valid_annotations"],
                3,
            )

            self.assertEqual(
                report["invalid_lines"],
                2,
            )

            self.assertEqual(
                report["files_scanned"],
                1,
            )

    def test_audit_yolo_source_summarizes_dataset(self):
        audit_yolo_source = self.require_api("audit_yolo_source")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "labels").mkdir(parents=True)
            (root / "valid" / "images").mkdir(parents=True)
            (root / "valid" / "labels").mkdir(parents=True)

            (root / "train" / "images" / "plastic.jpg").write_bytes(
                b"fake"
            )

            (root / "valid" / "images" / "glass.jpg").write_bytes(
                b"fake"
            )

            (root / "train" / "labels" / "plastic.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            (root / "valid" / "labels" / "glass.txt").write_text(
                "\n".join(
                    [
                        "2 0.4 0.4 0.2 0.2",
                        "9 0.5 0.5 0.2 0.2",
                    ]
                ),
                encoding="utf-8",
            )

            report = audit_yolo_source(
                root,
                nc=4,
            )

            self.assertEqual(report["images"], 2)
            self.assertEqual(report["label_files"], 2)

            self.assertEqual(
                report["image_splits"],
                {
                    "train": 1,
                    "valid": 1,
                },
            )

            self.assertEqual(
                report["label_splits"],
                {
                    "train": 1,
                    "valid": 1,
                },
            )

            self.assertEqual(
                report["class_counts"],
                {
                    0: 1,
                    2: 1,
                },
            )

            self.assertEqual(
                report["valid_annotations"],
                2,
            )

            self.assertEqual(
                report["invalid_lines"],
                1,
            )

    def test_audit_coco_source_summarizes_annotations(self):
        audit_coco_source = self.require_api("audit_coco_source")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            batch_dir = root / "batch_1"
            batch_dir.mkdir()

            (batch_dir / "plastic.jpg").write_bytes(
                b"fake"
            )

            annotations = {
                "images": [
                    {
                        "id": 1,
                        "file_name": "batch_1/plastic.jpg",
                        "width": 640,
                        "height": 480,
                    },
                    {
                        "id": 2,
                        "file_name": "batch_1/missing.jpg",
                        "width": 640,
                        "height": 480,
                    },
                ],
                "categories": [
                    {
                        "id": 4,
                        "name": "Clear plastic bottle",
                    },
                    {
                        "id": 8,
                        "name": "Food Can",
                    },
                ],
                "annotations": [
                    {
                        "id": 101,
                        "image_id": 1,
                        "category_id": 4,
                        "bbox": [10, 20, 100, 120],
                    },
                    {
                        "id": 102,
                        "image_id": 1,
                        "category_id": 4,
                        "bbox": [150, 100, 80, 90],
                    },
                    {
                        "id": 103,
                        "image_id": 2,
                        "category_id": 8,
                        "bbox": [30, 40, 50, 60],
                    },
                ],
            }

            annotations_path = root / "annotations.json"

            annotations_path.write_text(
                json.dumps(annotations),
                encoding="utf-8",
            )

            report = audit_coco_source(
                root,
                annotations_path,
            )

            self.assertEqual(
                report["images_declared"],
                2,
            )

            self.assertEqual(
                report["images_found"],
                1,
            )

            self.assertEqual(
                report["missing_images"],
                1,
            )

            self.assertEqual(
                report["annotations"],
                3,
            )

            self.assertEqual(
                report["categories"],
                2,
            )

            self.assertEqual(
                report["category_counts"],
                {
                    4: 2,
                    8: 1,
                },
            )

            self.assertEqual(
                report["category_names"],
                {
                    4: "Clear plastic bottle",
                    8: "Food Can",
                },
            )

    def test_audit_sources_writes_unified_json_report(self):
        audit_sources = self.require_api("audit_sources")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            yolo_root = root / "mini_yolo"

            (yolo_root / "train" / "images").mkdir(
                parents=True
            )
            (yolo_root / "train" / "labels").mkdir(
                parents=True
            )

            (
                yolo_root
                / "train"
                / "images"
                / "plastic.jpg"
            ).write_bytes(b"fake")

            (
                yolo_root
                / "train"
                / "labels"
                / "plastic.txt"
            ).write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            coco_root = root / "mini_coco"
            coco_root.mkdir()

            (coco_root / "glass.jpg").write_bytes(
                b"fake"
            )

            coco_annotations = {
                "images": [
                    {
                        "id": 1,
                        "file_name": "glass.jpg",
                        "width": 640,
                        "height": 480,
                    }
                ],
                "categories": [
                    {
                        "id": 6,
                        "name": "Glass bottle",
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 6,
                        "bbox": [10, 20, 100, 120],
                    }
                ],
            }

            annotations_path = (
                coco_root / "annotations.json"
            )

            annotations_path.write_text(
                json.dumps(coco_annotations),
                encoding="utf-8",
            )

            output_path = (
                root / "audit_report.json"
            )

            source_specs = {
                "mini_yolo": {
                    "format": "yolo",
                    "root": yolo_root,
                    "nc": 4,
                },
                "mini_coco": {
                    "format": "coco",
                    "root": coco_root,
                    "annotations": annotations_path,
                },
            }

            report = audit_sources(
                source_specs,
                output_path=output_path,
            )

            self.assertTrue(
                output_path.exists()
            )

            self.assertEqual(
                set(report["sources"]),
                {
                    "mini_yolo",
                    "mini_coco",
                },
            )

            self.assertEqual(
                report["sources"]["mini_yolo"]["images"],
                1,
            )

            self.assertEqual(
                report["sources"]["mini_yolo"][
                    "valid_annotations"
                ],
                1,
            )

            self.assertEqual(
                report["sources"]["mini_coco"][
                    "images_found"
                ],
                1,
            )

            self.assertEqual(
                report["sources"]["mini_coco"][
                    "annotations"
                ],
                1,
            )

            saved_report = json.loads(
                output_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                saved_report,
                report,
            )


if __name__ == "__main__":
    unittest.main()