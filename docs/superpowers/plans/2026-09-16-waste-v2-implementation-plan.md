# Waste V2 Dataset and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. All production behavior follows test-driven development: failing test first, then minimal implementation, then verification.

**Goal:** Build a clean, diverse, leakage-safe Waste V2 dataset and use it to fine-tune and evaluate YOLO26n V2.

**Architecture:** Candidate datasets are audited and normalized into the fixed four-class schema. A dedicated V2 builder excludes frozen benchmark images, validates annotations, groups augmentation families, removes duplicates, performs a group-aware train/validation split, writes provenance reports, and produces `data/processed/waste_v2`. Training then fine-tunes YOLO26n from the existing V1 `best.pt` and compares V2 against the frozen V1 external benchmark.

**Tech Stack:** Python 3.14, OpenCV, Ultralytics 8.4.142, PyTorch 2.14.0+cu126, YOLO26, pathlib, csv/json, hashlib.

**Spec:** `docs/superpowers/specs/2026-09-15-waste-v2-design.md`

## Global Constraints

- Final classes remain exactly:

  - 0 plastic

  - 1 metal

  - 2 glass

  - 3 paper_cardboard

- Reject is not a fifth YOLO class.

- Never train on `data/benchmarks/`.

- Never modify files under `data/raw/`.

- Existing V1 validation data remains validation-side.

- Roboflow augmentation siblings must not cross train/validation splits.

- Controlled negative images may have empty label files but must not dominate the dataset.

- All generated V2 data goes under `data/processed/waste_v2`.

- Training must not begin if benchmark leakage or split-family leakage is detected.

- V1 external benchmark remains the comparison baseline:

  - Precision 0.510

  - Recall 0.294

  - mAP50 0.273

  - mAP50-95 0.133

---

## Task 1: Audit All V2 Candidate Sources

**Files:**

- Create: `scripts/audit_waste_v2_sources.py`

- Create: `scripts/test_audit_waste_v2_sources.py`

- Output: `data/inspection/waste_v2_source_audit.json`

**Purpose:**

Confirm the real local state of every candidate source before building V2. The audit must count images, labels, classes, invalid labels, and source splits without modifying any dataset.

**Sources:**

- `data/processed/waste_v1`

- `data/raw/taco_optional`

- `data/raw/external_dataset_2/GARBAGE CLASSIFICATION`

- `data/raw/recyclable_waste_detection`

- `data/raw/conveyor_waste_belt`

- `data/benchmarks/external_test_v1`

- `data/benchmarks/reject_challenge_v1`

- [ ] Step 1: Write failing tests for source discovery, YOLO label validation, and class counting.

- [ ] Step 2: Run tests and verify they fail because the audit implementation does not yet exist.

- [ ] Step 3: Implement the minimum audit functions required by the tests.

- [ ] Step 4: Run audit tests until all pass.

- [ ] Step 5: Run the audit against real local datasets.

- [ ] Step 6: Inspect and save `waste_v2_source_audit.json`.

- [ ] Step 7: Commit the audit implementation and report.

---

## Task 2: Define V2 Class Mappings and Candidate Rules

**Files:**

- Create: `config/waste_v2_sources.json`

- Create: `scripts/test_waste_v2_mapping.py`

- Create or modify: `scripts/prepare_waste_v2.py`

**Required mappings:**

### Recyclable Waste Detection

- Cardboard -> 3

- Glass -> 2

- Metal -> 1

- Paper -> 3

- Plastic -> 0

### Conveyor Waste Belt

- Glass -> 2

- Metal-Other -> 1

- Paper-Cardboard -> 3

- Plastic -> 0

- Organic Food -> controlled negative candidate only

### Garbage Classification

- CARDBOARD -> 3

- GLASS -> 2

- METAL -> 1

- PAPER -> 3

- PLASTIC -> 0

- BIODEGRADABLE -> controlled negative candidate only

### TACO

Only explicitly approved recyclable categories may map to the final four classes. Ambiguous mixed images must be excluded when ignored objects would create false unlabeled regions.

- [ ] Step 1: Write failing mapping tests.

- [ ] Step 2: Verify expected failures.

- [ ] Step 3: Implement source configuration and mapping functions.

- [ ] Step 4: Run tests until green.

- [ ] Step 5: Commit mapping/configuration logic.

---

## Task 3: Frozen Benchmark Exclusion

**Files:**

- Modify: `scripts/prepare_waste_v2.py`

- Create: `scripts/test_waste_v2_benchmark_exclusion.py`

**Behavior:**

Before any candidate enters V2, compare it with the frozen benchmark using:

- exact content hash

- perceptual image hash

- source/provenance information when available

Any known benchmark match must be rejected.

The build report must count benchmark exclusions.

- [ ] Step 1: Write a failing test proving a benchmark duplicate is rejected.

- [ ] Step 2: Verify RED.

- [ ] Step 3: Implement exact hash exclusion.

- [ ] Step 4: Implement perceptual-hash exclusion.

- [ ] Step 5: Verify all exclusion tests pass.

- [ ] Step 6: Add an invariant: build fails if benchmark leakage remains.

- [ ] Step 7: Commit benchmark protection.

---

## Task 4: Duplicate and Family Grouping

**Files:**

- Modify: `scripts/prepare_waste_v2.py`

- Create: `scripts/test_waste_v2_grouping.py`

**Behavior:**

Create a stable family/group identifier for every accepted image.

Roboflow filenames containing augmentation identifiers such as `.rf.<hash>` must resolve to their original source family so sibling augmentations cannot be split across train and validation.

The builder must also:

- remove exact duplicate images

- prevent duplicate candidate insertion

- remove duplicated annotation lines

- [ ] Step 1: Write failing family-group tests.

- [ ] Step 2: Verify RED.

- [ ] Step 3: Implement family ID extraction.

- [ ] Step 4: Implement exact duplicate elimination.

- [ ] Step 5: Implement duplicate annotation-line removal.

- [ ] Step 6: Verify GREEN.

- [ ] Step 7: Commit grouping/dedup logic.

---

## Task 5: Annotation Validation

**Files:**

- Modify: `scripts/prepare_waste_v2.py`

- Create: `scripts/test_waste_v2_labels.py`

**Valid YOLO annotation requirements:**

- exactly 5 values

- class ID in 0..3

- x_center in valid normalized range

- y_center in valid normalized range

- width > 0

- height > 0

- normalized box remains valid

- image is readable

Invalid images or labels must be excluded and reported.

- [ ] Step 1: Write failing tests for malformed labels.

- [ ] Step 2: Verify RED.

- [ ] Step 3: Implement label parsing and validation.

- [ ] Step 4: Verify GREEN.

- [ ] Step 5: Commit validation logic.

---

## Task 6: Controlled Negative Samples

**Files:**

- Modify: `scripts/prepare_waste_v2.py`

- Create: `scripts/test_waste_v2_negatives.py`

**Behavior:**

Clean images containing only approved non-target material may be used as negative/background examples with empty YOLO label files.

Primary candidates:

- Organic Food from Conveyor Waste Belt

- selected BIODEGRADABLE-only examples from Garbage Classification

Negative images must be capped so they remain a minority of the final training set.

Mixed ambiguous images must not silently become negative samples.

- [ ] Step 1: Write failing negative-selection tests.

- [ ] Step 2: Verify RED.

- [ ] Step 3: Implement controlled negative eligibility.

- [ ] Step 4: Implement configurable negative cap.

- [ ] Step 5: Verify GREEN.

- [ ] Step 6: Commit controlled-negative logic.

---

## Task 7: Group-Aware Train / Validation Split

**Files:**

- Modify: `scripts/prepare_waste_v2.py`

- Create: `scripts/test_waste_v2_split.py`

**Target:**

Approximately:

- 88% train

- 12% validation

Rules:

- split by family/group, never individual augmented image

- preserve existing V1 validation data on validation side

- represent multiple source datasets in validation

- no family may exist in both splits

- [ ] Step 1: Write a failing family-leakage test.

- [ ] Step 2: Verify RED.

- [ ] Step 3: Implement deterministic group-aware splitting.

- [ ] Step 4: Add V1 validation preservation.

- [ ] Step 5: Verify train/validation family intersection is empty.

- [ ] Step 6: Commit split logic.

---

## Task 8: Build Waste V2

**Files produced:**

`data/processed/waste_v2/`

with:

- `images/train`

- `images/val`

- `labels/train`

- `labels/val`

- `data.yaml`

- `manifest.csv`

- `build_report.json`

**Manifest fields should include:**

- output image

- source dataset

- source path

- original split

- exact hash

- perceptual hash

- family ID

- final split

- negative flag

**Build report includes:**

- candidate count

- accepted count

- train count

- validation count

- counts by source

- counts by final class

- annotation counts

- negative count

- invalid files removed

- exact duplicates removed

- perceptual duplicates removed

- benchmark overlaps removed

- train/validation leakage status

- [ ] Step 1: Run all V2 builder tests.

- [ ] Step 2: Execute the real V2 build.

- [ ] Step 3: Inspect build statistics.

- [ ] Step 4: Verify benchmark leakage = 0.

- [ ] Step 5: Verify family leakage = 0.

- [ ] Step 6: Verify every label class is 0..3.

- [ ] Step 7: Verify representative images visually.

- [ ] Step 8: Freeze the processed V2 dataset for the first V2 training run.

---

## Task 9: YOLO26n V2 Fine-Tuning

**Files:**

- Create: `scripts/train_yolo26n_v2.py`

- Create: `docs/YOLO26N_TRAINING_V2.md`

**Starting weights:**

Existing V1:

`runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt`

**Dataset:**

`data/processed/waste_v2/data.yaml`

**Strategy:**

Fine-tune V1 rather than restarting blindly from zero.

GPU:

NVIDIA GeForce RTX 4050 Laptop GPU

Training settings must be recorded exactly in the V2 training document.

- [ ] Step 1: Verify CUDA.

- [ ] Step 2: Verify V2 YAML and class order.

- [ ] Step 3: Run a short smoke training run.

- [ ] Step 4: Confirm no dataset/configuration errors.

- [ ] Step 5: Start the full YOLO26n V2 training.

- [ ] Step 6: Save training evidence and best weights.

- [ ] Step 7: Commit training configuration/documentation.

---

## Task 10: Evaluate V2 Against V1

**Evaluation datasets:**

1\. Waste V2 validation

2\. Frozen 2500-image external benchmark

3\. Frozen 300-image Reject challenge

**Baseline to beat:**

V1 external:

- Precision 0.510

- Recall 0.294

- mAP50 0.273

- mAP50-95 0.133

- [ ] Step 1: Evaluate V2 validation.

- [ ] Step 2: Evaluate V2 on frozen external benchmark.

- [ ] Step 3: Compare V1/V2 overall metrics.

- [ ] Step 4: Compare per-class metrics.

- [ ] Step 5: Inspect confusion matrix and failure examples.

- [ ] Step 6: Test Reject challenge behavior.

- [ ] Step 7: Document whether V2 qualifies as the new detector.

---

## Task 11: Controlled Escalation if YOLO26n V2 Is Insufficient

Do not immediately collect another huge dataset or change multiple variables.

If YOLO26n V2 remains insufficient:

1\. inspect the weak classes/domains

2\. identify targeted data gaps

3\. consider YOLO26s using the same Waste V2 dataset

4\. compare against the exact same frozen benchmark

This keeps the experiment interpretable.

---

## Task 12: Resume Prototype Integration

Once detector quality is acceptable:

1\. Live Camera inference

2\. Decision / Reject logic

3\. PySerial

4\. Arduino communication

5\. Conveyor control

6\. Robotic arm pickup

7\. Sensors

8\. Integrated prototype

9\. Reliability testing

10\. Demo preparation

After 21 September, focus on reliability, fixes, integration stability, and demo readiness rather than adding unnecessary features.
