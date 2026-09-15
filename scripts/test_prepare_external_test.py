
import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prepare_external_test import (
    remap_dataset2_class,
    coco_bbox_to_yolo,
    choose_primary_class,
)

class PrepareExternalTestTests(unittest.TestCase):
    def test_dataset2_class_mapping(self):
        self.assertIsNone(remap_dataset2_class(0))  # biodegradable -> not a core detection class
        self.assertEqual(remap_dataset2_class(5), 0)  # plastic
        self.assertEqual(remap_dataset2_class(3), 1)  # metal
        self.assertEqual(remap_dataset2_class(2), 2)  # glass
        self.assertEqual(remap_dataset2_class(1), 3)  # cardboard
        self.assertEqual(remap_dataset2_class(4), 3)  # paper

    def test_coco_bbox_to_yolo(self):
        # x=10,y=20,w=40,h=20 on 100x100 image
        xc, yc, w, h = coco_bbox_to_yolo([10, 20, 40, 20], 100, 100)
        self.assertAlmostEqual(xc, 0.30)
        self.assertAlmostEqual(yc, 0.30)
        self.assertAlmostEqual(w, 0.40)
        self.assertAlmostEqual(h, 0.20)

    def test_primary_class_prefers_rarest_available_class(self):
        freq = {0: 100, 1: 40, 2: 80, 3: 120}
        self.assertEqual(choose_primary_class({0, 1, 3}, freq), 1)

if __name__ == "__main__":
    unittest.main()
