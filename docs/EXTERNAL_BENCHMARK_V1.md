# External Benchmark V1 and V2 Training Decision

Date: 2026-09-15

## Purpose

This document records the first external generalization benchmark of the Smart Recycle AI-Hub YOLO26n V1 model, the datasets used to build that benchmark, leakage/duplicate controls, the measured results, and the reason the project is moving to Training V2.

## V1 classes

- 0: plastic
- 1: metal
- 2: glass
- 3: paper_cardboard

Reject is decision logic, not a fifth YOLO class.

## V1 model

Model:
`runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt`

Training:
- YOLO26n
- 80/80 epochs completed
- best.pt selected

Environment:
- Python 3.14.6
- PyTorch 2.14.0+cu126
- CUDA: True
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB
- OpenCV 5.0.0
- Ultralytics 8.4.142

## Why an external benchmark was created

The University of Tehran Waste Management AI dataset was already used in V1 training, so it was not eligible as an external benchmark source.

The external benchmark was built from unseen sources instead.

## Source 1: TACO

Local source:
`data/raw/taco_optional`

Observed:
- 1500 images
- 4784 annotations
- 60 categories

TACO had not been used in V1 training.

After mapping eligible categories into the four V1 classes and excluding ambiguous mixed cases:
- clean core candidates: 997
- reject-only candidates: 65

## Source 2: Garbage Classification

Local source:
`data/raw/external_dataset_2/GARBAGE CLASSIFICATION`

Classes:
- 0 BIODEGRADABLE
- 1 CARDBOARD
- 2 GLASS
- 3 METAL
- 4 PAPER
- 5 PLASTIC

Observed:
- 10464 images / label files
- 74090 annotations
- 0 bad label lines

Annotation counts:
- BIODEGRADABLE: 45407
- CARDBOARD: 4698
- GLASS: 7809
- METAL: 5841
- PAPER: 4390
- PLASTIC: 5945

Images containing each class:
- BIODEGRADABLE: 2287
- CARDBOARD: 1502
- GLASS: 2684
- METAL: 1898
- PAPER: 1768
- PLASTIC: 1464

Mapping:
- CARDBOARD -> paper_cardboard
- GLASS -> glass
- METAL -> metal
- PAPER -> paper_cardboard
- PLASTIC -> plastic

BIODEGRADABLE was excluded from the four-class mAP benchmark and reserved for Reject/negative use.

After eligibility filtering:
- clean core candidates: 8177
- reject-only candidates: 2077

## External benchmark construction

Builder:
`scripts/prepare_external_test.py`

Target:
- 2500 external detection images
- 300 Reject challenge images

Before selection, the V1 processed dataset was scanned:
- images scanned: 16416
- unique perceptual dHashes: 16392
- unreadable: 0

Deduplication:
- clean usable external candidates after dedupe: 8955
- removed because they matched V1 training/validation: 187
- removed as duplicates inside candidate pool: 32
- unreadable: 0

Final benchmark:
- total images: 2500
- TACO: 470
- Garbage Classification: 2030

Images containing each final class:
- plastic: 695
- metal: 625
- glass: 851
- paper_cardboard: 692

Ground-truth annotations:
- plastic: 2394
- metal: 1881
- glass: 2499
- paper_cardboard: 1829
- total: 8603

Reject challenge:
- total: 300
- dataset2_reject: 288
- taco_reject: 12

## Frozen benchmark

After evaluation the datasets were moved locally to:

- `data/benchmarks/external_test_v1`
- `data/benchmarks/reject_challenge_v1`

Local protection marker:
`data/benchmarks/DO_NOT_TRAIN.txt`

These images must never be added to training.

The 2500-image set is now a fixed development benchmark for comparing V1/V2/V3. Because its results have already been inspected, a new unseen dataset must later be used as the final blind test.

The benchmark images themselves remain local and are not intended for Git storage.

## YOLO26n V1 external benchmark results

Overall:

| Metric | Result |
|---|---:|
| Precision | 0.510 |
| Recall | 0.294 |
| mAP50 | 0.273 |
| mAP50-95 | 0.133 |

Per class:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| plastic | 695 | 2394 | 0.507 | 0.305 | 0.260 | 0.126 |
| metal | 625 | 1881 | 0.589 | 0.291 | 0.300 | 0.146 |
| glass | 851 | 2499 | 0.445 | 0.288 | 0.237 | 0.104 |
| paper_cardboard | 692 | 1829 | 0.499 | 0.293 | 0.295 | 0.154 |

Performance:
- preprocess: 2.5 ms/image
- inference: 4.1 ms/image
- postprocess: 1.7 ms/image

Interpretation:
- inference speed is already strong
- generalization and recall are not strong enough
- glass was the weakest class on mAP50-95
- V1 remains the baseline but is not accepted as the final detector

## Decision: build Training V2

V2 will keep the benchmark frozen and build a larger, cleaner, more diverse training dataset.

Candidate sources already available locally:

### Existing V1 data
`data/processed/waste_v1`
Observed size: 16416 images

### Remaining TACO + Garbage Classification candidates
Thousands of eligible images remain outside the frozen 2500-image benchmark.

### Recyclable Waste Detection
Observed:
- 6396 images
- classes: Cardboard, Glass, Metal, Paper, Plastic
- license metadata: CC BY 4.0

### Conveyor Waste Belt
Observed:
- 1798 images
- classes: Glass, Metal-Other, Organic Food, Paper-Cardboard, Plastic
- license metadata: CC BY 4.0

Organic Food will not become a fifth class. Clean organic-only examples may be used as controlled negative/background samples.

## V2 safety rules

The V2 builder must:
1. never use frozen benchmark images
2. never modify raw datasets in place
3. map all target classes to the same four-class schema
4. validate all YOLO annotations
5. remove exact duplicate annotation lines
6. remove exact/perceptual duplicate images
7. prevent Roboflow augmentation siblings from crossing train/valid splits
8. preserve provenance/source metadata
9. use controlled negatives without allowing them to dominate
10. write a new dataset under `data/processed/waste_v2`

## Planned model progression

YOLO26n V1 baseline
-> build waste_v2
-> fine-tune YOLO26n V2
-> evaluate on frozen 2500 benchmark
-> if still insufficient, train YOLO26s on the same V2 data
-> later fine-tune on real conveyor/camera/lighting images
-> final acceptance on a new blind external test

Development quality target for the controlled conveyor scenario:
- Precision around or above 0.80
- Recall around or above 0.80

These are targets, not guaranteed results.

## Current checkpoint

DONE:
- Dataset V1 prepared
- YOLO26n V1 training 80/80
- best.pt
- Streamlit inference workflow
- external benchmark builder
- 2500-image external benchmark
- 300-image Reject challenge
- V1 external evaluation
- benchmark frozen from training

NEXT:
- Waste V2 dataset construction
- YOLO26n V2 training
- YOLO26s escalation if needed

AFTER MODEL QUALITY GATE:
- live camera
- decision / Reject
- PySerial / Arduino
- conveyor / arm / sensor integration
