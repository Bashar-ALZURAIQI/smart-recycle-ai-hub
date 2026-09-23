import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class TestBuildWasteV2(unittest.TestCase):

    def test_family_split_never_leaks_across_train_and_valid(self):
        builder = importlib.import_module(
            "scripts.build_waste_v2"
        )

        split_function = getattr(
            builder,
            "split_records_by_family",
            None,
        )

        self.assertIsNotNone(
            split_function,
            "split_records_by_family() is missing",
        )

        records = [
            {
                "id": "a1",
                "family_id": "family_a",
                "source": "waste_v1",
            },
            {
                "id": "a2",
                "family_id": "family_a",
                "source": "waste_v1",
            },
            {
                "id": "b1",
                "family_id": "family_b",
                "source": "garbage_classification",
            },
            {
                "id": "b2",
                "family_id": "family_b",
                "source": "garbage_classification",
            },
            {
                "id": "c1",
                "family_id": "family_c",
                "source": "recyclable_waste_detection",
            },
            {
                "id": "d1",
                "family_id": "family_d",
                "source": "conveyor_waste_belt",
            },
        ]

        train_records, valid_records = (
            split_function(
                records,
                valid_fraction=0.25,
                seed=26,
            )
        )

        train_families = {
            record["family_id"]
            for record in train_records
        }

        valid_families = {
            record["family_id"]
            for record in valid_records
        }

        self.assertTrue(
            train_records
        )

        self.assertTrue(
            valid_records
        )

        self.assertTrue(
            train_families.isdisjoint(
                valid_families
            ),
            "A family leaked across train and valid",
        )

        combined_ids = {
            record["id"]
            for record in (
                train_records
                + valid_records
            )
        }

        expected_ids = {
            record["id"]
            for record in records
        }

        self.assertEqual(
            combined_ids,
            expected_ids,
        )

    def test_validation_keeps_all_sources_represented_when_possible(
        self,
    ):
        builder = importlib.import_module(
            "scripts.build_waste_v2"
        )

        records = []

        sources = [
            "waste_v1",
            "garbage_classification",
            "recyclable_waste_detection",
            "conveyor_waste_belt",
        ]

        for source in sources:
            for family_number in range(4):
                records.append(
                    {
                        "id": (
                            f"{source}_"
                            f"{family_number}"
                        ),
                        "family_id": (
                            f"{source}_family_"
                            f"{family_number}"
                        ),
                        "source": source,
                    }
                )

        train_records, valid_records = (
            builder.split_records_by_family(
                records,
                valid_fraction=0.25,
                seed=26,
            )
        )

        train_sources = {
            record["source"]
            for record in train_records
        }

        valid_sources = {
            record["source"]
            for record in valid_records
        }

        expected_sources = set(
            sources
        )

        self.assertEqual(
            train_sources,
            expected_sources,
            (
                "Every source should remain "
                "represented in train"
            ),
        )

        self.assertEqual(
            valid_sources,
            expected_sources,
            (
                "Every source should be "
                "represented in valid "
                "when enough families exist"
            ),
        )

    def test_controlled_negatives_are_capped_and_deterministic(
        self,
    ):
        builder = importlib.import_module(
            "scripts.build_waste_v2"
        )

        select_function = getattr(
            builder,
            "select_controlled_negatives",
            None,
        )

        self.assertIsNotNone(
            select_function,
            "select_controlled_negatives() is missing",
        )

        positive_records = [
            {
                "id": f"positive_{number}",
                "is_negative": False,
            }
            for number in range(8)
        ]

        negative_records = [
            {
                "id": f"negative_{number}",
                "is_negative": True,
            }
            for number in range(10)
        ]

        all_records = (
            positive_records
            + negative_records
        )

        first_selection = (
            select_function(
                all_records,
                max_negative_fraction=0.20,
                seed=26,
            )
        )

        second_selection = (
            select_function(
                all_records,
                max_negative_fraction=0.20,
                seed=26,
            )
        )

        selected_negatives = [
            record
            for record in first_selection
            if record["is_negative"]
        ]

        selected_positives = [
            record
            for record in first_selection
            if not record["is_negative"]
        ]

        self.assertEqual(
            len(selected_positives),
            8,
            "Positive records must never be discarded",
        )

        self.assertLessEqual(
            len(selected_negatives),
            2,
            (
                "Controlled negatives should not exceed "
                "20% of the final selected dataset"
            ),
        )

        self.assertEqual(
            first_selection,
            second_selection,
            (
                "Negative selection must be deterministic "
                "for the same seed"
            ),
        )

    def test_write_dataset_output_creates_yolo_structure(
        self,
    ):
        builder = importlib.import_module(
            "scripts.build_waste_v2"
        )

        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(
                temp_directory
            )

            source_directory = (
                root
                / "source"
            )

            source_directory.mkdir()

            train_positive_image = (
                source_directory
                / "train_positive.jpg"
            )

            train_negative_image = (
                source_directory
                / "train_negative.jpg"
            )

            valid_positive_image = (
                source_directory
                / "valid_positive.jpg"
            )

            train_positive_image.write_bytes(
                b"train-positive"
            )

            train_negative_image.write_bytes(
                b"train-negative"
            )

            valid_positive_image.write_bytes(
                b"valid-positive"
            )

            train_records = [
                {
                    "id": "train_positive",
                    "family_id": "family_1",
                    "source": "waste_v1",
                    "image_path": train_positive_image,
                    "label_text": (
                        "0 0.5 0.5 0.2 0.2\n"
                    ),
                    "is_negative": False,
                },
                {
                    "id": "train_negative",
                    "family_id": "family_2",
                    "source": "conveyor_waste_belt",
                    "image_path": train_negative_image,
                    "label_text": "",
                    "is_negative": True,
                },
            ]

            valid_records = [
                {
                    "id": "valid_positive",
                    "family_id": "family_3",
                    "source": "garbage_classification",
                    "image_path": valid_positive_image,
                    "label_text": (
                        "3 0.4 0.4 0.3 0.3\n"
                    ),
                    "is_negative": False,
                },
            ]

            output_directory = (
                root
                / "waste_v2"
            )

            builder.write_dataset_output(
                train_records,
                valid_records,
                output_directory,
            )

            self.assertTrue(
                (
                    output_directory
                    / "train"
                    / "images"
                    / "train_positive.jpg"
                ).exists()
            )

            self.assertTrue(
                (
                    output_directory
                    / "train"
                    / "images"
                    / "train_negative.jpg"
                ).exists()
            )

            self.assertTrue(
                (
                    output_directory
                    / "valid"
                    / "images"
                    / "valid_positive.jpg"
                ).exists()
            )

            train_positive_label = (
                output_directory
                / "train"
                / "labels"
                / "train_positive.txt"
            )

            train_negative_label = (
                output_directory
                / "train"
                / "labels"
                / "train_negative.txt"
            )

            valid_positive_label = (
                output_directory
                / "valid"
                / "labels"
                / "valid_positive.txt"
            )

            self.assertEqual(
                train_positive_label.read_text(
                    encoding="utf-8"
                ),
                "0 0.5 0.5 0.2 0.2\n",
            )

            self.assertEqual(
                train_negative_label.read_text(
                    encoding="utf-8"
                ),
                "",
            )

            self.assertEqual(
                valid_positive_label.read_text(
                    encoding="utf-8"
                ),
                "3 0.4 0.4 0.3 0.3\n",
            )

            data_yaml = (
                output_directory
                / "data.yaml"
            )

            self.assertTrue(
                data_yaml.exists()
            )

            yaml_text = data_yaml.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "train: train/images",
                yaml_text,
            )

            self.assertIn(
                "val: valid/images",
                yaml_text,
            )

            self.assertIn(
                "plastic",
                yaml_text,
            )

            self.assertIn(
                "metal",
                yaml_text,
            )

            self.assertIn(
                "glass",
                yaml_text,
            )

            self.assertIn(
                "paper_cardboard",
                yaml_text,
            )

    def test_write_dataset_output_creates_manifest_and_report(
        self,
    ):
        builder = importlib.import_module(
            "scripts.build_waste_v2"
        )

        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(
                temp_directory
            )

            source_directory = (
                root
                / "source"
            )

            source_directory.mkdir()

            image_a = (
                source_directory
                / "image_a.jpg"
            )

            image_b = (
                source_directory
                / "image_b.jpg"
            )

            image_c = (
                source_directory
                / "image_c.jpg"
            )

            image_a.write_bytes(
                b"a"
            )

            image_b.write_bytes(
                b"b"
            )

            image_c.write_bytes(
                b"c"
            )

            train_records = [
                {
                    "id": "record_a",
                    "family_id": "family_a",
                    "source": "waste_v1",
                    "image_path": image_a,
                    "label_text": (
                        "0 0.5 0.5 0.2 0.2\n"
                    ),
                    "is_negative": False,
                },
                {
                    "id": "record_b",
                    "family_id": "family_b",
                    "source": "conveyor_waste_belt",
                    "image_path": image_b,
                    "label_text": "",
                    "is_negative": True,
                },
            ]

            valid_records = [
                {
                    "id": "record_c",
                    "family_id": "family_c",
                    "source": "garbage_classification",
                    "image_path": image_c,
                    "label_text": (
                        "3 0.4 0.4 0.3 0.3\n"
                    ),
                    "is_negative": False,
                },
            ]

            output_directory = (
                root
                / "waste_v2"
            )

            builder.write_dataset_output(
                train_records,
                valid_records,
                output_directory,
            )

            manifest_path = (
                output_directory
                / "manifest.csv"
            )

            report_path = (
                output_directory
                / "build_report.json"
            )

            self.assertTrue(
                manifest_path.exists(),
                "manifest.csv was not created",
            )

            self.assertTrue(
                report_path.exists(),
                "build_report.json was not created",
            )

            with manifest_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as manifest_file:

                rows = list(
                    csv.DictReader(
                        manifest_file
                    )
                )

            self.assertEqual(
                len(rows),
                3,
            )

            manifest_by_id = {
                row["id"]: row
                for row in rows
            }

            self.assertEqual(
                manifest_by_id[
                    "record_a"
                ][
                    "split"
                ],
                "train",
            )

            self.assertEqual(
                manifest_by_id[
                    "record_b"
                ][
                    "is_negative"
                ],
                "True",
            )

            self.assertEqual(
                manifest_by_id[
                    "record_c"
                ][
                    "split"
                ],
                "valid",
            )

            report = json.loads(
                report_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                report[
                    "train_records"
                ],
                2,
            )

            self.assertEqual(
                report[
                    "valid_records"
                ],
                1,
            )

            self.assertEqual(
                report[
                    "total_records"
                ],
                3,
            )

            self.assertEqual(
                report[
                    "positive_records"
                ],
                2,
            )

            self.assertEqual(
                report[
                    "negative_records"
                ],
                1,
            )

            self.assertEqual(
                report[
                    "sources"
                ][
                    "waste_v1"
                ],
                1,
            )

            self.assertEqual(
                report[
                    "sources"
                ][
                    "conveyor_waste_belt"
                ],
                1,
            )

            self.assertEqual(
                report[
                    "sources"
                ][
                    "garbage_classification"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()