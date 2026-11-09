# Waste V2.1 Dataset and Training Design

Date: 2026-09-23

## 1. Purpose

Waste V2.1 is a targeted data-and-fine-tuning iteration for Smart Recycle AI-Hub.

The goal is to improve real-world material recognition while preserving the four-class detector architecture and the frozen evaluation benchmarks.

Final detector classes remain exactly:

- `0 plastic`

- `1 metal`

- `2 glass`

- `3 paper_cardboard`

`Reject` remains decision-layer logic and is not a YOLO class.

V2 remains the baseline.

V2.1 starts from the selected V2 checkpoint:

`runs/detect/waste_v2_yolo26n/weights/best.pt`

The existing processed V2 dataset remains unchanged and is used as an input source:

`data/processed/waste_v2`

Raw V2.1 sources are read-only and are never modified in place.

---

## 2. Protected Evaluation Data

The following roots are permanently excluded from V2.1 training and validation:

- `data/benchmarks/external_test_v1`

- `data/benchmarks/reject_challenge_v1`

- `data/benchmarks/real_world_v2_1`

Every candidate source, including the existing V2 dataset, must be screened against these protected benchmark roots.

Protection includes exact SHA-256 checks and perceptual duplicate checks.

No protected image may survive in either final train or final validation.

An empty `real_world_v2_1` directory is valid and must not break the build.

---

## 3. Final Detection Classes

The detector continues to use exactly four classes:

- `0 = plastic`

- `1 = metal`

- `2 = glass`

- `3 = paper_cardboard`

These class IDs must remain unchanged throughout V2.1.

`Reject` is not a fifth YOLO class.

Reject will continue to be handled later by the decision layer using confidence, detection state, and system logic.

---

## 4. Logical Data Sources

Waste V2.1 combines four logical sources:

1. Existing Waste V2 dataset

2. WhiteMind YOLO-Waste Detection v1

3. Waste Detection Dataset 1 v3

4. General Waste Data v4

All source-provided train, valid, and test folders are treated as candidate pools.

V2.1 creates its own final family-aware train/validation split after filtering and deduplication.

Final output:

`data/processed/waste_v2_1`

---

## 5. Existing Waste V2

Root:

`data/processed/waste_v2`

Waste V2 remains unchanged.

Observed V2 build totals:

- Final images: 32,292

- Train images: 28,423

- Validation images: 3,869

- Positive images: 30,184

- Controlled negatives: 2,108

- Family overlap: 0

Mapping is identity mapping:

- `0 -> 0 plastic`

- `1 -> 1 metal`

- `2 -> 2 glass`

- `3 -> 3 paper_cardboard`

Existing V2 controlled negatives remain eligible for V2.1.

V2.1 must not create new negatives by stripping unsupported objects from the three new external datasets.

---

## 6. Source 1 — WhiteMind YOLO-Waste Detection v1

Root:

`data/raw/v2_1_sources/whitemind_yolo_waste`

Downloaded metadata:

- Version: 1

- License: CC BY 4.0

Original classes:

- `0 Biodegradable - Compost`

- `1 Biodegradable | Compost`

- `2 Glass`

- `3 Metal`

- `4 Mixed`

- `5 Paper`

- `6 Plastic`

Mapping:

- `2 -> 2 glass`

- `3 -> 1 metal`

- `5 -> 3 paper_cardboard`

- `6 -> 0 plastic`

Unsupported classes:

- `0`

- `1`

- `4`

Policy:

- Keep only images whose annotations are entirely from supported classes.

- If any unsupported class occurs in an image, exclude the whole image.

- Never remove only the unsupported box while keeping the image.

- Do not convert unsupported-only images into negatives.

Audited candidate pool:

- Original images: 2,808

- Supported-only images: 2,772

- Kept annotations: 8,675

- Plastic annotations: 2,566

- Metal annotations: 1,985

- Glass annotations: 1,504

- Paper/cardboard annotations: 2,620

Quality audit:

- Bad format: 0

- Bad class IDs: 0

- Bad coordinates: 0

- Zero-size boxes: 0

---

## 7. Source 2 — Waste Detection Dataset 1 v3

Root:

`data/raw/v2_1_sources/waste_detection_dataset_1`

Downloaded metadata:

- Version: 3

- License: CC BY 4.0

Original dataset:

- Images: 17,182

- Annotations: 27,803

- Empty label files: 91

- Classes: 30

Supported mappings:

### Plastic -> 0

- `11 HDPE Bottles`

- `20 PET Bottle`

- `21 PET Cup`

- `22 PS Plastic`

- `25 Plastic Bag`

- `26 Plastic Wrapper`

- `29 plastic straw`

### Metal -> 1

- `0 Aluminium Foil`

- `15 Metal Cans`

- `16 Metal Scraps`

### Glass -> 2

- `1 Broken Glass`

- `9 Glass Bottle`

- `10 Glass Jars`

### Paper/Cardboard -> 3

- `4 Cardboard`

- `14 Magazines`

- `18 Newspaper`

- `19 Office Paper`

- `24 Paper`

Unsupported classes:

- `2 Bulbs`

- `3 Cables`

- `5 Charger`

- `6 Cylindrical battery`

- `7 Earphone`

- `8 Food Waste`

- `12 Headphone`

- `13 Kitchen Waste`

- `17 Mobile Phone`

- `23 Paint Containers`

- `27 Pouch battery`

- `28 facemask`

Policy:

- Keep supported-only images.

- Exclude supported-plus-unsupported images entirely.

- Exclude unsupported-only images entirely.

- Exclude empty-label images.

- Exclude the whole image if any annotation line is not standard five-token YOLO detection format.

- Never rewrite or repair the raw polygon annotations.

Audit before non-five-token filtering:

- Supported-only images: 9,574

- Supported annotations: 13,482

- Plastic: 6,527

- Paper/cardboard: 2,889

- Metal: 2,482

- Glass: 1,584

Annotation-format audit:

- Non-five-token lines: 325

- Files affected: 205

- Supported-only images affected: 36

Those 36 supported-only images are excluded entirely.

Duplicate audit:

- Images hashed: 17,182

- Unique SHA-256 hashes: 17,170

- Exact duplicate hash groups across train and valid: 4

Image/label consistency:

- Images without labels: 0

- Labels without images: 0

---

## 8. Source 3 — General Waste Data v4

Root:

`data/raw/v2_1_sources/general_waste_data`

Downloaded metadata:

- Version: 4

- License: CC BY 4.0

Unsupported classes:

- `0 BIODEGRADABLE`

- `2 CLOTH`

Supported mapping:

### Paper/Cardboard -> 3

- `1 CARDBOARD`

- `21 PAPER`

- `22 PAPER cup`

### Glass -> 2

- `3 GLASS`

- `4 GLASS bottle`

- `5 GLASS bowl`

- `6 GLASS broken`

- `7 GLASS cap`

- `8 GLASS jar`

- `9 GLASS plate`

- `10 GLASS water`

### Metal -> 1

- `11 METAL`

- `12 METAL Jar`

- `13 METAL cane`

- `14 METAL cap`

- `15 METAL knife`

- `16 METAL paper`

- `17 METAL plate`

- `18 METAL scissor`

- `19 METAL sheet`

- `20 METAL spoon`

### Plastic -> 0

- `23 PLASTIC`

- `24 PLASTIC bag`

- `25 PLASTIC bottle`

- `26 PLASTIC can`

- `27 PLASTIC cane`

- `28 PLASTIC cap`

- `29 PLASTIC cup`

- `30 PLASTIC plate`

- `31 PLASTIC wrapper`

Policy:

- Keep supported-only images.

- Exclude the entire image if BIODEGRADABLE or CLOTH occurs.

- Exclude empty-label images.

Audited candidate pool:

- Original images: 17,788

- Original annotations: 125,561

- Supported-only images: 13,897

- Kept annotations: 46,915

- Paper/cardboard: 14,816

- Glass: 12,798

- Plastic: 9,917

- Metal: 9,384

Quality audit:

- Bad format: 0

- Bad class IDs: 0

- Bad coordinates: 0

- Zero-size boxes: 0

Duplicate audit:

- Unique SHA-256 hashes: 17,788

- Exact duplicate copies: 0

- Cross-split exact duplicates: 0

---

## 9. Mapping Configuration

Source mappings and filtering rules live in:

`config/waste_v2_1_mapping.yaml`

The configuration is the source of truth for:

- Source ID

- Source root

- Source type

- Class mapping

- Unsupported classes

- Empty-label policy

- Non-detection-label policy

- Source priority

- Benchmark roots

- Split seed

- Validation fraction

Mapping decisions must not be independently duplicated in several unrelated code paths.

---

## 10. Builder

Primary builder:

`scripts/build_waste_v2_1.py`

The build must be deterministic for fixed inputs, configuration, and seed.

Pipeline order:

1. Validate configuration.

2. Discover source image/label pairs.

3. Parse annotations without changing raw files.

4. Apply source-level filtering.

5. Apply class remapping.

6. Create deterministic family IDs.

7. Apply controlled-negative policy to existing V2 negatives.

8. Fingerprint candidates.

9. Fingerprint protected benchmark images.

10. Remove benchmark leakage.

11. Remove exact duplicates.

12. Remove perceptual duplicates.

13. Keep all members of one family in one final split.

14. Create final family-aware train/validation split.

15. Copy selected images and write remapped labels.

16. Generate `data.yaml`.

17. Generate manifest.

18. Generate machine-readable build report.

19. Run final integrity checks.

20. Fail the build if any invariant is violated.

The implementation may reuse proven logic from `scripts/build_waste_v2.py`, but existing V2 code and generated V2 data must remain unchanged.

---

## 11. Exact and Perceptual Deduplication

Exact duplicates use SHA-256.

The preferred duplicate source priority is:

1. `waste_v2`

2. `general_waste_data`

3. `whitemind_yolo_waste`

4. `waste_detection_dataset_1`

Stable path ordering is the tie-breaker inside one source.

Perceptual duplicate identity uses:

- dHash

- Image width

- Image height

Perceptual key:

`(dhash, width, height)`

An exact matching perceptual key is treated as a perceptual duplicate.

No exact or perceptual duplicate may survive in both final train and validation.

---

## 12. Families

Roboflow filenames containing `.rf.` are grouped using the stem before `.rf.`.

Existing augmentation suffix normalization from Waste V2 is retained.

Family IDs include source identity.

Every member of one family must remain in the same final split.

No family ID may appear in both train and validation.

Final split seed:

`26`

Target validation fraction:

`0.12`

---

## 13. Controlled Negatives

Only existing V2 controlled negative images remain eligible.

New external-source unsupported objects are not converted into negatives.

Maximum negative fraction:

`0.20`

If the available V2 negatives exceed the configured maximum, the builder deterministically selects the allowed subset using seed 26.

---

## 14. Output Layout

Required output:

```text

data/processed/waste_v2_1/

&#x20; data.yaml

&#x20; manifest.csv

&#x20; build_report.json

&#x20; train/

&#x20;   images/

&#x20;   labels/

&#x20; valid/

&#x20;   images/

&#x20;   labels/

```

Output filenames must be deterministic and collision-resistant.

Manifest provenance must contain:

- source

- source split

- family ID

- negative flag

- original image path

- output name

- final split

- SHA-256

- dHash

- width

- height

---

## 15. Build Invariants

A successful build must satisfy all of the following:

- Final class IDs are only `{0,1,2,3}`.

- Each non-empty output label line contains exactly five YOLO tokens.

- Coordinates are normalized to `[0,1]`.

- Width and height are greater than zero.

- Every output image has one label file.

- Every output label has one image.

- Controlled negative labels may be empty.

- No protected benchmark exact duplicate survives.

- No protected benchmark perceptual duplicate survives.

- No exact duplicate survives across final train and validation.

- No family ID appears in both train and validation.

- Unsupported annotations are never silently converted into background.

- Fixed inputs and seed produce deterministic selection and splitting.

The builder exits non-zero if a final invariant fails.

---

## 16. Testing

Tests live in:

`tests/test_build_waste_v2_1.py`

Required coverage includes:

- Config validation

- V2 identity mapping

- WhiteMind mapping

- WhiteMind whole-image exclusion

- Dataset 1 unsupported-image exclusion

- Dataset 1 empty-label exclusion

- Dataset 1 non-five-token exclusion

- General Waste Data mapping

- General Waste Data BIODEGRADABLE/CLOTH exclusion

- Deterministic exact duplicate winner

- Protected benchmark exclusion

- Empty benchmark directory safety

- Roboflow family normalization

- Family-safe splitting

- Seed-26 determinism

- Controlled negative cap

- Output class/geometry validation

The full existing project test suite must remain green.

---

## 17. Dataset Report

After the real build create:

`docs/waste_v2_1_dataset_report.md`

It records actual generated values for:

- Candidate images by source

- Exclusions by source and reason

- Non-five-token exclusions

- Exact duplicate removals

- Perceptual duplicate removals

- Benchmark removals

- Final train count

- Final validation count

- Positive image count

- Negative image count

- Final annotation counts

- Final class distribution

- Final source distribution

- Family overlap result

- Benchmark leakage result

Audit estimates in this design document do not replace the final generated report.

---

## 18. Training

Training begins only after the dataset build and full sanity checks pass.

Training script:

`scripts/train_waste_v2_1.py`

Starting checkpoint:

`runs/detect/waste_v2_yolo26n/weights/best.pt`

Training never starts from V1 and never starts from the pilot checkpoint.

Pilot configuration:

- Full Waste V2.1 training split

- 1 epoch

- Image size 640

- Batch 8 initially

- CUDA required

- Workers 0

- Seed 26

- AMP enabled

If and only if CUDA out-of-memory occurs, batch 4 may be tried once.

Full V2.1 fine-tuning configuration:

- Start from V2 `best.pt`

- Maximum epochs: 11

- Patience: 3

- Image size: 640

- Optimizer: auto

- Batch: pilot-selected 8 or 4

- Seed: 26

- Deterministic mode enabled

- AMP enabled

- Workers: 0

- Cache: false

The pilot is used to verify real epoch duration, VRAM behavior, dataset correctness, and training stability.

The full run restarts from the original V2 checkpoint.

---

## 19. Training Completion

Training completion requires:

- `runs/detect/waste_v2_1_yolo26n/weights/best.pt`

- `runs/detect/waste_v2_1_yolo26n/weights/last.pt`

- `runs/detect/waste_v2_1_yolo26n/results.csv`

The selected `best.pt` must successfully load with Ultralytics.

The loaded model class names must be exactly:

- plastic

- metal

- glass

- paper_cardboard

The training-completion report is:

`docs/waste_v2_1_training_report.md`

---

## 20. Git and Reproducibility

Work remains on:

`feature/waste-v2.1`

Raw datasets are not committed.

Generated processed datasets are not committed unless repository policy explicitly changes.

Training run directories and weight files are not committed as ordinary Git blobs.

Tracked items include:

- Design documentation

- Implementation plan

- Mapping configuration

- Builder

- Tests

- Dataset report

- Training script

- Training report

Waste V2 and Waste V2 history must remain unchanged.

When V2.1 is eventually integrated into main, preserve the individual V2.1 commits rather than squash-merging if contribution history is intended to remain visible.

---

## 21. Completion Criteria

The V2.1 dataset phase is complete when:

- All four logical sources pass through one reproducible builder.

- Mapping matches this design.

- Unsupported and ambiguous images follow the defined source policies.

- Benchmark leakage is zero.

- Exact/perceptual duplicate checks pass.

- Family overlap is zero.

- Final labels contain only the four project classes.

- Build report is generated.

- Builder tests pass.

- Full project test suite passes.

The V2.1 training phase is complete when:

- One full-dataset pilot epoch succeeds.

- Full fine-tuning starts from V2 `best.pt`.

- Full training completes naturally or by early stopping.

- V2.1 `best.pt` exists.

- V2.1 `last.pt` exists.

- `results.csv` exists.

- `best.pt` loads successfully.

- Model class names are correct.
