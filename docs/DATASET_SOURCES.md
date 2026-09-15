# Smart Recycle AI-Hub - Dataset Sources

## Final V1/V2 Classes

0 = plastic
1 = metal
2 = glass
3 = paper_cardboard

Reject is NOT a training class.

---

## Primary V1 Training Dataset

Name:
Waste Management AI

Provider:
University of Tehran / Roboflow Universe

Task:
Object Detection

License:
CC BY 4.0

Classes:
- Paper
- Plastic
- Glass
- Metal
- Cardboard

Mapping:
Plastic -> plastic
Metal -> metal
Glass -> glass
Paper -> paper_cardboard
Cardboard -> paper_cardboard

Status:
USED IN V1 TRAINING

Notes:
This dataset must NOT be used as an external unseen benchmark for V1 because it contributed to the V1 training dataset.

---

## Secondary V1 Dataset

Name:
DWSD - Dense Waste Segmentation Dataset

Task:
Segmentation

License:
CC BY 4.0

Approved Mapping:
plastic containers -> plastic
plastic bottles -> plastic
plastic -> plastic
plastic cups -> plastic
metal bottles -> metal
glass -> glass
paper -> paper_cardboard

Other classes:
EXCLUDE / REVIEW

Status:
V1 SECONDARY SOURCE

---

## External Benchmark Source 1

Name:
TACO - Trash Annotations in Context

Task:
Detection / Segmentation

Annotation Format:
COCO

Annotation License:
CC BY 4.0

Local Path:
data/raw/taco_optional

Observed Locally:
- 1500 images
- 4784 annotations
- 60 categories

Status:
USED FOR V1 EXTERNAL BENCHMARK CANDIDATES

Notes:
TACO was NOT used in the V1 training dataset.
Eligible target categories were mapped to the four Smart Recycle classes.
Ambiguous/unknown-only examples were separated for Reject testing where appropriate.

---

## External Benchmark Source 2

Name:
Garbage Classification

Task:
Object Detection

Annotation Format:
YOLO

Local Path:
data/raw/external_dataset_2/GARBAGE CLASSIFICATION

Observed Locally:
- 10464 images
- 74090 annotations
- 0 bad label lines

Source Classes:
0 = BIODEGRADABLE
1 = CARDBOARD
2 = GLASS
3 = METAL
4 = PAPER
5 = PLASTIC

Mapping:
CARDBOARD -> paper_cardboard
GLASS -> glass
METAL -> metal
PAPER -> paper_cardboard
PLASTIC -> plastic

BIODEGRADABLE:
Excluded from the four-class mAP benchmark and reserved for negative / Reject use.

Status:
USED FOR V1 EXTERNAL BENCHMARK CANDIDATES

---

## Frozen Development Benchmark

Local Paths:
- data/benchmarks/external_test_v1
- data/benchmarks/reject_challenge_v1

Observed:
- 2500 core benchmark images
- 300 Reject challenge images

Status:
FROZEN - NEVER USE FOR TRAINING

Purpose:
Compare V1, V2, V3, and later development models on the same fixed data.

Important:
Because the benchmark results have already been inspected, it is a development benchmark rather than the final blind test.
A separate unseen final test dataset will be acquired later.

---

## V2 Candidate Dataset 1

Name:
Recyclable Waste Detection

Provider:
Roboflow Universe

Task:
Object Detection

License:
CC BY 4.0

Local Path:
data/raw/recyclable_waste_detection

Observed Locally:
- 6396 images

Classes:
- Cardboard
- Glass
- Metal
- Paper
- Plastic

Planned Mapping:
Cardboard -> paper_cardboard
Glass -> glass
Metal -> metal
Paper -> paper_cardboard
Plastic -> plastic

Status:
DOWNLOADED - V2 TRAINING CANDIDATE

---

## V2 Candidate Dataset 2

Name:
Conveyor Waste Belt

Provider:
Roboflow Universe

Task:
Object Detection

License:
CC BY 4.0

Local Path:
data/raw/conveyor_waste_belt

Observed Locally:
- 1798 images

Classes:
- Glass
- Metal-Other
- Organic Food
- Paper-Cardboard
- Plastic

Planned Mapping:
Glass -> glass
Metal-Other -> metal
Paper-Cardboard -> paper_cardboard
Plastic -> plastic

Organic Food:
Do NOT map to a fifth YOLO class.
Clean organic-only images may be used as controlled negative/background samples.

Status:
DOWNLOADED - V2 TRAINING CANDIDATE

---

## V2 Data Safety Rules

1. Never train on images in data/benchmarks/.
2. Never modify raw datasets in place.
3. Build a new processed dataset under data/processed/waste_v2.
4. Use exact and perceptual duplicate checks before inclusion.
5. Prevent Roboflow augmentation siblings from leaking across train/validation splits.
6. Keep all final YOLO target labels in the four-class schema only.
7. Preserve dataset/source provenance in V2 manifests.
8. Reserve a new unseen dataset for the final blind test.
