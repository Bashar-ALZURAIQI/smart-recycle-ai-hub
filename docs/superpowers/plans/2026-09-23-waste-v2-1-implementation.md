# Waste V2.1 Dataset Build and Training Implementation Plan

## Goal

Build a reproducible Waste V2.1 dataset from Waste V2 plus three audited external datasets, verify that the final dataset is clean and benchmark-safe, then fine-tune YOLO26n from the selected Waste V2 checkpoint.

## Starting Point

Branch:

`feature/waste-v2.1`

Current remote baseline before V2.1 work:

`0238518 merge: preserve Waste V2 commit history`

Starting model:

`runs/detect/waste_v2_yolo26n/weights/best.pt`

Final classes:

- `0 plastic`

- `1 metal`

- `2 glass`

- `3 paper_cardboard`

Reject remains decision-layer logic.

---

## Global Rules

- Never modify raw datasets in place.

- Never modify `data/processed/waste_v2`.

- Never train on benchmark images.

- Never create a fifth YOLO class.

- Never silently strip unsupported annotations from a mixed image.

- New external-source unsupported images are excluded rather than used as negatives.

- Existing V2 controlled negatives may remain eligible.

- All final dataset construction must be deterministic.

- Seed is 26.

- Validation target is approximately 12% by family.

- Maximum controlled negative fraction is 20%.

- Exact dedupe uses SHA-256.

- Perceptual dedupe uses exact `(dhash, width, height)` identity.

- Roboflow augmentation families must not cross train/valid.

- Training starts from Waste V2 `best.pt`.

---

## Task 1 — Documentation and Mapping Contract

Create:

- `docs/superpowers/specs/2026-09-23-waste-v2-1-design.md`

- `docs/superpowers/plans/2026-09-23-waste-v2-1-implementation.md`

- `docs/waste_v2_1_source_audit.md`

- `config/waste_v2_1_mapping.yaml`

Verify:

- YAML parses successfully.

- All four source IDs exist.

- Protected benchmark roots are configured.

- Source priority is configured.

- Seed and validation fraction are correct.

Commit:

`docs: define Waste V2.1 dataset contract`

Push the checkpoint to:

`origin/feature/waste-v2.1`

---

## Task 2 — Source Filtering and Mapping

Create:

- `scripts/build_waste_v2_1.py`

- `tests/test_build_waste_v2_1.py`

Implement:

- Configuration loading

- Source policy loading

- Strict five-token YOLO detection parser

- Whole-image unsupported-class exclusion

- Empty-label policy

- Class remapping

- Roboflow family normalization

- Source discovery

- Image/label pairing

Required tests:

- WhiteMind supported mapping

- WhiteMind mixed-image exclusion

- Dataset 1 non-five-token exclusion

- New-source empty-label exclusion

- V2 negative preservation

- Roboflow `.rf.` family grouping

- Source discovery using temporary fixtures

Commit:

`feat: add Waste V2.1 source filtering`

---

## Task 3 — Benchmark Protection and Deduplication

Add to builder:

- SHA-256 image fingerprint

- dHash image fingerprint

- Width/height fingerprint metadata

- Protected benchmark fingerprint collection

- Exact benchmark exclusion

- Perceptual benchmark exclusion

- Deterministic exact dedupe

- Deterministic perceptual dedupe

Source priority:

1. waste_v2

2. general_waste_data

3. whitemind_yolo_waste

4. waste_detection_dataset_1

Tie-breaker:

Stable image-path ordering.

Required tests:

- Protected exact match removed

- Empty benchmark directory is safe

- Duplicate winner follows source priority

- Renamed identical image does not survive twice

- Perceptual-key duplicate does not survive twice

Commit:

`feat: protect V2.1 from leakage and duplicates`

---

## Task 4 — Controlled Negatives and Family-Safe Split

Implement:

- Controlled V2 negative selection

- Maximum negative fraction 0.20

- Deterministic seed 26

- Family-aware train/valid split

- Validation source representation where practical

- Zero family overlap

Required tests:

- Negative fraction cap

- Family never crosses train/valid

- Split is deterministic

- Multiple-source validation representation

- Single-family edge case

Commit:

`feat: add family-safe V2.1 splitting`

---

## Task 5 — Final Dataset Writer and Build Report

Implement:

- Deterministic collision-resistant output names

- `train/images`

- `train/labels`

- `valid/images`

- `valid/labels`

- `data.yaml`

- `manifest.csv`

- `build_report.json`

- Final output validation

- Main build orchestration

- CLI

Final output:

`data/processed/waste_v2_1`

Manifest columns:

- source

- source_split

- family_id

- is_negative

- source_image

- output_name

- final_split

- sha256

- dhash

- width

- height

Final validation must check:

- Class IDs only 0–3

- Five-token non-empty labels

- Valid normalized coordinates

- Positive width and height

- Image/label count equality

- Zero benchmark exact overlap

- Zero benchmark perceptual overlap

- Zero family overlap

- Zero invalid classes

Required end-to-end fixture test must verify:

- Mixed unsupported image is excluded

- Protected image is excluded

- Roboflow family does not cross splits

- Output files exist

- Final report has zero invariant failures

Commit:

`feat: build reproducible Waste V2.1 dataset`

---

## Task 6 — Real Dataset Build

Before build:

- Verify working tree

- Verify all raw source roots exist

- Verify Waste V2 exists

- Verify benchmark roots

Run:

`python scripts\build_waste_v2_1.py`

Expected output:

`data/processed/waste_v2_1`

Inspect:

- Build report

- Train images

- Valid images

- Positive images

- Negative images

- Class distribution

- Source distribution

- Dedupe removals

- Benchmark removals

- Family overlap

Create:

`docs/waste_v2_1_dataset_report.md`

Run:

- Builder test suite

- Full project test suite

Verify raw/generated data are not accidentally staged.

Commit:

`docs: record Waste V2.1 dataset build`

Push pre-training checkpoint to:

`origin/feature/waste-v2.1`

---

## Task 7 — Training Workflow

Create:

- `scripts/train_waste_v2_1.py`

- `tests/test_train_waste_v2_1.py`

Training dataset:

`data/processed/waste_v2_1/data.yaml`

Starting checkpoint:

`runs/detect/waste_v2_yolo26n/weights/best.pt`

Implement preflight:

- Dataset exists

- Train image/label counts match

- Valid image/label counts match

- Non-empty labels have exactly five tokens

- Classes are only 0–3

- Coordinates are valid

- Every class exists in train

- Every class exists in valid

Implement GPU verification:

- CUDA must be available

- CPU fallback is forbidden

- Print GPU name

- Print available total VRAM

Pilot:

- 1 epoch

- Full training split

- Batch 8

- Image size 640

- Workers 0

- Cache false

- Seed 26

- Deterministic true

- AMP true

- Optimizer auto

- Output name `waste_v2_1_pilot`

Full:

- Maximum 11 epochs

- Patience 3

- Batch selected by pilot

- Image size 640

- Workers 0

- Cache false

- Seed 26

- Deterministic true

- AMP true

- Optimizer auto

- Output name `waste_v2_1_yolo26n`

Pilot and full run both start from the original Waste V2 `best.pt`.

Do not start the full run from pilot weights.

Required tests:

- Preflight class counts

- Invalid label rejection

- Pilot argument selection

- Full argument selection

Commit:

`feat: add Waste V2.1 training workflow`

Push checkpoint to:

`origin/feature/waste-v2.1`

---

## Task 8 — Pilot and Full Training

First run all tests.

Confirm starting checkpoint:

`runs/detect/waste_v2_yolo26n/weights/best.pt`

Run pilot:

`python scripts\train_waste_v2_1.py --mode pilot --batch 8`

If and only if CUDA OOM occurs:

- Close GPU-heavy programs.

- Check `nvidia-smi`.

- Retry once with batch 4.

Do not change image size or model.

Record:

- Pilot wall time

- Epoch time

- Validation time

- Selected batch

- GPU memory usage if reported

- Loss behavior

If pilot succeeds with batch 8:

`python scripts\train_waste_v2_1.py --mode full --batch 8`

If pilot requires batch 4:

`python scripts\train_waste_v2_1.py --mode full --batch 4`

Full training settings:

- Start checkpoint: Waste V2 best.pt

- Maximum epochs: 11

- Patience: 3

- Optimizer: auto

- Image size: 640

- Seed: 26

- Deterministic: true

- AMP: true

- Workers: 0

- Cache: false

Required outputs:

- `runs/detect/waste_v2_1_yolo26n/weights/best.pt`

- `runs/detect/waste_v2_1_yolo26n/weights/last.pt`

- `runs/detect/waste_v2_1_yolo26n/results.csv`

Verify:

- `best.pt` loads with Ultralytics.

- Model class names are correct.

Create:

`docs/waste_v2_1_training_report.md`

Record only observed training values.

Run full tests again.

Commit:

`docs: record Waste V2.1 training`

Push final training-complete branch.

---

## Final Training Completion Gate

V2.1 training is complete only when:

- Dataset builder passes.

- Benchmark leakage is zero.

- Family overlap is zero.

- Final classes are correct.

- Full project test suite passes.

- Pilot succeeds.

- Full training completes.

- `best.pt` exists.

- `last.pt` exists.

- `results.csv` exists.

- `best.pt` loads successfully.

- Training report is committed.
