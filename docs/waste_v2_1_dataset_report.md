# Waste V2.1 Dataset Build Report

Date: 2026-09-25

## 1. Overview

Waste V2.1 was built successfully from the existing Waste V2 dataset plus three audited external YOLO datasets.

Builder:

`python scripts\build_waste_v2_1.py`

Builder implementation commit:

`96d6def feat: build reproducible Waste V2.1 dataset`

Configuration:

`config/waste_v2_1_mapping.yaml`

Generated dataset:

`data/processed/waste_v2_1`

The detector continues to use exactly four classes:

- `0 = plastic`
- `1 = metal`
- `2 = glass`
- `3 = paper_cardboard`

`Reject` remains decision-layer logic and is not a YOLO class.

---

## 2. Build Configuration

The real dataset build used the approved Waste V2.1 configuration:

| Setting | Value |
|---|---:|
| Seed | 26 |
| Validation target | 0.12 |
| Maximum controlled-negative fraction | 0.20 |
| Exact duplicate identity | SHA-256 |
| Perceptual duplicate identity | Exact `(dhash, width, height)` |
| Final detector classes | 4 |

Protected benchmark roots:

- `data/benchmarks/external_test_v1`
- `data/benchmarks/reject_challenge_v1`
- `data/benchmarks/real_world_v2_1`

---

## 3. Source Filtering Summary

The filtered candidate pool contained exactly 58,499 images.

The per-source candidate counts below come from the audited source filtering contract and sum exactly to the builder-reported candidate total.

| Source | Original Images | Candidate Images | Excluded Before Build Pool |
|---|---:|---:|---:|
| Waste V2 | 32,292 | 32,292 | 0 |
| WhiteMind YOLO-Waste | 2,808 | 2,772 | 36 |
| Waste Detection Dataset 1 | 17,182 | 9,538 | 7,644 |
| General Waste Data | 17,788 | 13,897 | 3,891 |
| **Total** | **70,070** | **58,499** | **11,571** |

### WhiteMind YOLO-Waste

Original images:

`2,808`

Candidate images:

`2,772`

Excluded:

`36`

The excluded images contained unsupported classes.

Supported mapping:

- Glass -> `2`
- Metal -> `1`
- Paper -> `3`
- Plastic -> `0`

Unsupported classes were excluded at whole-image level.

No unsupported annotations were stripped while keeping the image.

### Waste Detection Dataset 1

Original images:

`17,182`

Candidate images:

`9,538`

Total excluded:

`7,644`

Audited exclusion breakdown:

- Unsupported or mixed-class images: `7,517`
- Empty-label images: `91`
- Supported-only images rejected because of non-five-token annotations: `36`

The raw polygon/non-detection annotations were not repaired or converted.

### General Waste Data

Original images:

`17,788`

Candidate images:

`13,897`

Total excluded:

`3,891`

Audited exclusion breakdown:

- Mixed supported/unsupported images: `353`
- Unsupported-only images: `3,537`
- Empty-label images: `1`

Unsupported classes were BIODEGRADABLE and CLOTH.

### Existing Waste V2

All `32,292` existing Waste V2 images remained eligible.

Its `2,108` controlled negatives remained eligible for V2.1.

Waste V2 itself was not modified.

---

## 4. Candidate Pool

Builder-reported candidate records:

`58,499`

This exactly matches the sum of the four filtered source candidate pools:

`32,292 + 2,772 + 9,538 + 13,897 = 58,499`

---

## 5. Controlled Negatives

Controlled negatives available from Waste V2:

`2,108`

Controlled negatives kept:

`2,108`

Controlled negatives removed by the 20% cap:

`0`

No new negative examples were created from unsupported external-source images.

Final negative-image count:

`2,108`

Final positive-image count:

`54,047`

Final dataset size:

`56,155`

Final negative fraction:

approximately `3.75%`

This is below the configured maximum of `20%`.

---

## 6. Benchmark Protection

Protected benchmark screening was applied before final dataset output.

Observed removals:

| Benchmark Match Type | Removed |
|---|---:|
| Exact SHA-256 | 0 |
| Exact perceptual `(dhash, width, height)` | 0 |

A removal count of zero means none of the surviving candidate images matched the protected benchmark fingerprints.

Final invariant validation also reported zero benchmark leakage.

---

## 7. Deduplication

Observed duplicate removals:

| Duplicate Type | Removed |
|---|---:|
| Exact SHA-256 duplicates | 1,274 |
| Perceptual duplicates | 1,070 |
| **Total duplicate removals** | **2,344** |

The arithmetic reconciles exactly:

`58,499 - 1,274 - 1,070 = 56,155`

Source priority used for duplicate winner selection:

1. `waste_v2`
2. `general_waste_data`
3. `whitemind_yolo_waste`
4. `waste_detection_dataset_1`

Stable source-path ordering was used as the tie-breaker inside a source.

---

## 8. Final Dataset Size

Final records:

`56,155`

Train:

`49,416`

Validation:

`6,739`

Observed validation fraction:

approximately `12.00%`

This matches the configured target of approximately `12%`.

Positive images:

`54,047`

Negative images:

`2,108`

---

## 9. Final Source Distribution

| Source | Final Images |
|---|---:|
| Waste V2 | 32,292 |
| General Waste Data | 12,888 |
| Waste Detection Dataset 1 | 9,468 |
| WhiteMind YOLO-Waste | 1,507 |
| **Total** | **56,155** |

The source totals reconcile exactly with the final manifest size.

---

## 10. Final Annotation Distribution

| Class ID | Class | Annotation Count |
|---:|---|---:|
| 0 | plastic | 43,160 |
| 1 | metal | 32,595 |
| 2 | glass | 40,346 |
| 3 | paper_cardboard | 64,597 |
| **Total** |  | **180,698** |

All final annotation class IDs are within the required `{0, 1, 2, 3}` set.

---

## 11. Image and Label Pairing

Manifest rows:

`56,155`

Train images:

`49,416`

Train labels:

`49,416`

Validation images:

`6,739`

Validation labels:

`6,739`

Therefore:

- Train image/label counts match.
- Validation image/label counts match.
- Every final manifest record corresponds to one generated image and one generated label file.

Empty label files are permitted only for controlled negative images.

---

## 12. Final Integrity Validation

The generated `build_report.json` reported:

`"invariant_failures": []`

The successful final validation therefore found no violations of the configured Waste V2.1 invariants.

Validated conditions include:

- Final class IDs are only `0–3`.
- Non-empty labels use five-token YOLO detection format.
- Bounding-box coordinates are normalized.
- Bounding-box width and height are positive.
- Train image and label files are paired.
- Validation image and label files are paired.
- No protected benchmark exact overlap survived.
- No protected benchmark perceptual overlap survived.
- No exact train/validation duplicate overlap survived.
- No perceptual train/validation duplicate overlap survived.
- No family ID crosses train and validation.

Family overlap result:

`0`

Benchmark leakage result:

`0`

Invariant failures:

`0`

---

## 13. Generated Dataset Layout

The final dataset exists at:

`data/processed/waste_v2_1`

Generated structure:

```text
data/processed/waste_v2_1/
  data.yaml
  manifest.csv
  build_report.json

  train/
    images/
    labels/

  valid/
    images/
    labels/
```

Generated dataset files remain outside Git staging.

---

## 14. Verification

Waste V2.1-specific test suite after the real build:

`32 passed, 30 deselected`

Full project test suite after the real build:

`89 passed`

Additional unittest subtests:

`42 passed`

Warnings:

`4`

The four warnings originate from the older Waste V2 implementation:

`scripts/prepare_waste_v2.py:592`

They are Pillow deprecation warnings for `Image.Image.getdata()` and are not Waste V2.1 build failures.

`git diff --check` completed without errors.

`git status --short` was empty before creation of this report, confirming that generated dataset files were not accidentally staged.

---

## 15. Build Result

Waste V2.1 real dataset build completed successfully.

Final status:

- Candidate images: `58,499`
- Final images: `56,155`
- Train images: `49,416`
- Validation images: `6,739`
- Positive images: `54,047`
- Controlled negatives: `2,108`
- Exact duplicates removed: `1,274`
- Perceptual duplicates removed: `1,070`
- Benchmark removals: `0`
- Final annotations: `180,698`
- Family overlap: `0`
- Benchmark leakage: `0`
- Invariant failures: `0`

The generated Waste V2.1 dataset is ready for the training-workflow stage.