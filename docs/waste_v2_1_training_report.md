# Smart Recycle AI-Hub — YOLO26n Waste V2.1 Training and External Benchmark Report

**Report finalized:** 2026-11-10  
**Branch:** `feature/waste-v2.1`  
**Model status:** **Selected as the improved development candidate**; final blind and physical-system tests remain pending.  
**Scope:** Four-class waste object detection, not physical sorting accuracy.

## 1. Experiment objective and model lineage

Waste V2.1 was developed as an evidence-based fine-tuning iteration of the official Waste V2 YOLO26n checkpoint. The model preserves four detector classes:

| ID | Class |
|---:|---|
| 0 | `plastic` |
| 1 | `metal` |
| 2 | `glass` |
| 3 | `paper_cardboard` |

`Reject` is **not** a YOLO training class. Unknown/unsupported objects and uncertain detections require a separate decision/reject policy.

The **full V2.1 run started from** `runs/detect/waste_v2_yolo26n/weights/best.pt`. It did **not** start from randomly initialized weights or from the one-epoch V2.1 pilot.

## 2. V2.1 dataset provenance and controls

These figures come from the version-controlled dataset build report, `docs/waste_v2_1_dataset_report.md` (2026-09-25):

| Item | Recorded result |
|---|---:|
| Final images | 56,155 |
| Training images | 49,416 |
| Validation images | 6,739 |
| Positive images | 54,047 |
| Controlled negative images | 2,108 |
| Total object annotations | 180,698 |
| Exact duplicates removed | 1,274 |
| Perceptual duplicates removed | 1,070 |
| Reported train/validation family overlap | 0 |
| Reported protected benchmark overlap | 0 |
| Reported build invariant failures | 0 |

The build combined existing Waste V2 with three audited external YOLO datasets, applying class remapping, format checks, deduplication, provenance tracking and family-aware splitting.

The dataset builder's checks reported no exact/perceptual benchmark overlap under its defined fingerprint methods. This is **not** proof that every possible semantic near-duplicate or source dependency was absent.

The following benchmark roots were protected from training and internal validation:

- `data/benchmarks/external_test_v1`
- `data/benchmarks/reject_challenge_v1`
- `data/benchmarks/real_world_v2_1`

## 3. Training setup

Run metadata were read from local `args.yaml` and `results.csv`, using the read-only training audit performed on 2026-11-09.

| Setting | V2.1 full run |
|---|---|
| Network | YOLO26n |
| Source checkpoint | `runs/detect/waste_v2_yolo26n/weights/best.pt` |
| Dataset | `data/processed/waste_v2_1/data.ultralytics.yaml` |
| Run name | `waste_v2_1_yolo26n` |
| Configured epochs | 11 |
| Logged epochs | 11 |
| Image size | 640 |
| Batch | 8 |
| Optimizer configuration | `auto` |
| `lr0` | 0.01 |
| `lrf` | 0.01 |
| Patience | 3 |
| Seed | 26 |
| GPU device | 0 (CUDA) |
| Workers | 0 |
| AMP / deterministic / cache | Enabled / enabled / disabled per training-script configuration |

**Important:** `optimizer=auto` is the **configured mode**, not independently verified proof of which optimizer Ultralytics ultimately selected. The actual optimizer and training duration need full training logs if they are to be reported.

The environment reported during the later benchmark evaluation was Python 3.14.6, PyTorch 2.14.0+cu126, Ultralytics 8.4.142 and an NVIDIA GeForce RTX 4050 Laptop GPU (6140 MiB). Those versions are directly verified for **evaluation**; the training environment has not been independently reconstructed from full logs here.

## 4. Pilot and full training outcomes

| Run | Logged epochs | Best logged *internal-validation* mAP50-95 |
|---|---:|---:|
| `waste_v2_1_pilot` | 1 | 0.4388 |
| `waste_v2_1_yolo26n` | 11 | 0.6447 |

The pilot served as a feasibility and stability check. The full experiment restarted from the official V2 checkpoint. Local file modification times place the pilot output on 2026-09-25 and the latest full-run `results.csv` update on 2026-10-02 (Türkiye local time). File timestamps do not independently establish all training start/stop times.

### Best logged internal validation — full V2.1

The highest logged mAP50-95 occurred at **epoch 11**, the final configured epoch:

| Metric | Value |
|---|---:|
| Precision | 0.8580 |
| Recall | 0.7287 |
| mAP50 | 0.8239 |
| mAP50-95 | 0.6447 |
| Training box loss | 0.7176 |
| Training classification loss | 0.8466 |

The following local artifacts were confirmed present:

- `runs/detect/waste_v2_1_yolo26n/results.csv`
- `runs/detect/waste_v2_1_yolo26n/args.yaml`
- `runs/detect/waste_v2_1_yolo26n/weights/best.pt`
- `runs/detect/waste_v2_1_yolo26n/weights/last.pt`

The 11 recorded epochs match the **maximum of 11** specified in the project design. No claim is made here that `best.pt` and `last.pt` are byte-identical.

**Internal validation must not be compared naively across versions:** Waste V2 used 3,869 internal validation images; Waste V2.1 used 6,739. An identical external benchmark is required for a credible comparison.

## 5. Frozen external development benchmark: protocol

**Evaluation date:** 2026-11-10 (Türkiye local time).  
**Dataset:** `data/benchmarks/external_test_v1`  
**Images:** 2,500  
**Annotated objects:** 8,603  
**Evaluated model weights:** V2 `best.pt` and V2.1 `best.pt`.

Both models were re-evaluated in the **same session** and with identical settings:

- `imgsz=640`
- `batch=8`
- `device=0`
- `workers=0`
- `split='test'`
- Ultralytics YOLO validation; no training was performed.

The saved local evaluation outputs are:

- V2: `runs/benchmark_v2_1_comparison/v2_reference/`
- V2.1: `runs/benchmark_v2_1_comparison/v2_1_candidate/`
- Summary: `runs/benchmark_v2_1_comparison/benchmark_comparison.md`
- Machine-readable results: `runs/benchmark_v2_1_comparison/benchmark_comparison.json`

The comparison was produced with the locally executed read-only `evaluate_v2_1_benchmark.py` utility. As of this report, that helper is **local and not asserted to be version-controlled**. Retain the original utility and JSON output with the experiment archive for exact reproducibility.

## 6. External benchmark results — V2 versus V2.1

| Metric | V2 | V2.1 | Change (V2.1 − V2) |
|---|---:|---:|---:|
| Precision | 0.5918 | 0.7630 | +0.1712 |
| Recall | 0.3847 | 0.5556 | +0.1709 |
| mAP50 | 0.4236 | 0.6491 | +0.2255 |
| mAP50-95 | 0.2813 | 0.4640 | +0.1826* |

*The mAP50-95 delta was produced using the original unrounded metric values; subtraction of the displayed four-decimal figures can differ by 0.0001 due to rounding.*

Relative to V2, mAP50 increased by approximately **53.2%** and mAP50-95 by approximately **64.9%**. These relative changes are not percentages of correct physical sorting.

The evaluator reported:

| Measurement | V2 | V2.1 |
|---|---:|---:|
| Preprocessing per image | 0.3 ms | 0.3 ms |
| Model inference per image | 3.8 ms | 3.8 ms |
| Postprocessing per image | 0.7 ms | 0.8 ms |

These are YOLO validation-stage timing observations on the benchmark machine, **not** total camera-to-robot latency or a guaranteed throughput rate.

## 7. Per-class external benchmark results

| Class | V2 mAP50 | V2.1 mAP50 | V2 mAP50-95 | V2.1 mAP50-95 | mAP50-95 change |
|---|---:|---:|---:|---:|---:|
| plastic | 0.3673 | 0.5693 | 0.2325 | 0.3813 | +0.1488 |
| metal | 0.3618 | 0.6380 | 0.2549 | 0.4420 | +0.1871 |
| glass | 0.4987 | 0.7438 | 0.3279 | 0.5573 | +0.2294 |
| paper_cardboard | 0.4666 | 0.6452 | 0.3101 | 0.4752 | +0.1651 |

The validation log additionally reported these V2.1 class-level precision/recall values (rounded to three decimals):

| Class | Precision | Recall |
|---|---:|---:|
| plastic | 0.756 | 0.466 |
| metal | 0.716 | 0.564 |
| glass | 0.837 | 0.655 |
| paper_cardboard | 0.742 | 0.537 |

All four classes improved in both mAP50 and mAP50-95. Glass had the largest absolute mAP50-95 gain (+0.2294); plastic retained the lowest mAP50-95 (0.3813) and recall (0.466) in the V2.1 result.

## 8. Interpretation and constraints

**Supported conclusion:** On this fixed 2,500-image **development** benchmark, the saved V2.1 checkpoint outperformed the selected V2 checkpoint on overall precision, recall, mAP50, mAP50-95, and per-class mAP.

**Not supported by these results:**

- Claiming 64.91% end-to-end classification or robotic sorting accuracy. mAP is an object-detection metric.
- Claiming V2.1 is fully ready for deployment in a physical sorting prototype.
- Claiming a previously examined development benchmark is a new, final blind test.
- Claiming the final reject policy safely handles unsupported material.
- Inferring error categories, confidence calibration, class confusion, or true physical throughput without the necessary targeted tests.

The development benchmark has been used across earlier project iterations. It is useful for comparisons, but its known results may have influenced development decisions, and it must **not** be portrayed as untouched final holdout evidence.

## 9. Model decision

**Decision:** Adopt V2.1 `best.pt` as the **preferred development/evaluation candidate** for subsequent error analysis and physical integration tests, while preserving the official V2 checkpoint for regression comparisons.

**Final acceptance remains pending**, subject to:

1. A genuinely unseen, properly separated final test set with controlled provenance.
2. Error review: missed objects, wrong-class predictions, false positives, glass/plastic confusion, reflections, small objects, difficult lighting/backgrounds.
3. Separate testing of unknown-object `Reject` decisions, confidence thresholds and multi-object handling.
4. Real camera / conveyor / sensor / robot-arm tests measuring end-to-end sorting success, misses, rejection errors and timing.

No retraining is justified solely by these positive benchmark results. Further work should be driven by documented failures.

## 10. Repository and evidence preservation

Version-controlled sources:

- `docs/superpowers/specs/2026-09-23-waste-v2-1-design.md`
- `docs/superpowers/plans/2026-09-23-waste-v2-1-implementation.md`
- `docs/waste_v2_1_source_audit.md`
- `docs/waste_v2_1_dataset_report.md`
- `config/waste_v2_1_mapping.yaml`
- `scripts/build_waste_v2_1.py`
- `scripts/train_waste_v2_1.py`

Local experiment evidence, currently excluded by `.gitignore`, must be backed up separately: `runs/`, `*.pt`, processed datasets, original benchmark data and the benchmark comparison JSON. Do **not** push the 2,500 benchmark images, raw datasets, or large model checkpoints as ordinary Git blobs.

**Milestone outcome:** Waste V2.1 fine-tuning and same-benchmark V2 comparison completed. Improved development model identified; external final blind/physical acceptance not yet completed.
