# YOLO26n V2 Training

## 1. Purpose

This document records the completed YOLO26n Waste V2 training and model-selection process for the Smart Recycle AI-Hub prototype.

The detector supports exactly four trainable classes:

```text
0: plastic
1: metal
2: glass
3: paper_cardboard
```

Reject is not a YOLO class.

Unsupported, unknown, ambiguous, or low-confidence objects will be handled later by Python decision logic.

---

## 2. Starting Model

Waste V2 was fine-tuned from the official Waste V1 checkpoint:

```text
runs/detect/runs/waste_v1/yolo26n_v1_baseline/weights/best.pt
```

V2 was not trained from random initialization.

---

## 3. Waste V2 Dataset

Dataset configuration:

```text
data/processed/waste_v2/data.yaml
```

Training split:

```text
Images:       28,423
Annotations: 101,939
Backgrounds:   1,855
```

Validation split:

```text
Images:       3,869
Annotations: 14,128
Backgrounds:     253
```

Classes:

```text
plastic
metal
glass
paper_cardboard
```

The dataset combines:

```text
Waste V1
Garbage Classification
Recyclable Waste Detection
Conveyor Waste Belt
```

TACO was not included in the fast Waste V2 build.

The frozen external benchmark was excluded from training.

---

## 4. Dataset Validation

Waste V2 preparation included:

- class remapping
- annotation validation
- image/label validation
- exact duplicate filtering
- perceptual duplicate filtering
- frozen benchmark exclusion
- family-aware train/validation splitting
- negative/background sample handling
- source provenance tracking

Final automated validation result:

```text
27 / 27 tests passed
```

Detailed dataset documentation:

```text
docs/DATASET_PREPARATION_V2.md
```

---

## 5. Training Environment

```text
Python:      3.14.6
Ultralytics: 8.4.142
PyTorch:     2.14.0+cu126
CUDA:        True
GPU:         NVIDIA GeForce RTX 4050 Laptop GPU
VRAM:        approximately 6 GB
```

Automatic Mixed Precision was enabled.

---

## 6. Windows Memory Stability

The first sanity attempt used:

```text
automatic batch
workers=4
```

Training completed, but validation later failed with an OpenCV/DataLoader memory allocation error.

This was CPU/RAM pressure and not a CUDA out-of-memory error.

The stable Windows configuration became:

```text
batch=8
workers=0
```

This configuration was retained for subsequent V2 training.

---

## 7. Sanity Training

Sanity configuration:

```text
Starting weights: V1 best.pt
Epochs:           1
Train fraction:   0.05
Image size:       640
Batch:            8
Workers:          0
Device:           CUDA:0
AMP:              True
```

Ultralytics used approximately:

```text
1,421 training images
3,869 validation images
```

Sanity result:

| Metric | Result |
|---|---:|
| Precision | 0.223 |
| Recall | 0.367 |
| mAP50 | 0.197 |
| mAP50-95 | 0.105 |

Per-class sanity result:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| plastic | 0.213 | 0.364 | 0.186 | 0.0929 |
| metal | 0.133 | 0.542 | 0.175 | 0.0919 |
| glass | 0.191 | 0.166 | 0.118 | 0.0629 |
| paper_cardboard | 0.356 | 0.395 | 0.309 | 0.171 |

The sanity metrics were not treated as final model performance.

Sanity status:

```text
PASS
```

---

## 8. Full V2 Experiment — MuSGD

The first full V2 experiment started from the original V1 best checkpoint.

Configuration:

```text
Image size:      640
Batch:           8
Workers:         0
Device:          CUDA:0
AMP:             True
Seed:            26
Deterministic:   True
Cache:           False
Maximum epochs:  80
Patience:        20
```

Ultralytics automatically selected:

```text
Optimizer: MuSGD
Initial LR: 0.01
Momentum: 0.9
```

The run directory was:

```text
runs/detect/waste_v2_yolo26n
```

---

## 9. MuSGD Internal Validation Results

The first four completed epochs were:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.809 | 0.658 | 0.756 | 0.601 |
| 2 | 0.804 | 0.656 | 0.748 | 0.594 |
| 3 | 0.776 | 0.624 | 0.716 | 0.552 |
| 4 | 0.781 | 0.623 | 0.717 | 0.560 |

The corresponding training-loss trend was:

```text
Epoch 1
box_loss = 0.69746
cls_loss = 1.16225
l1_loss  = 0.01094

Epoch 2
box_loss = 0.70645
cls_loss = 1.20256
l1_loss  = 0.01093

Epoch 3
box_loss = 0.75055
cls_loss = 1.33297
l1_loss  = 0.01167

Epoch 4
box_loss = 0.78082
cls_loss = 1.44972
l1_loss  = 0.01225
```

The best internal validation result occurred very early.

Approximate epoch durations increased significantly:

```text
Epoch 1: about 25 minutes training + validation
Epoch 2: about 40 minutes training + validation
Epoch 3: about 40 minutes training + validation
Epoch 4: about 45 minutes training + validation
```

Epoch 5 started, but the overall validation trend was already weaker than Epoch 1.

The experiment was manually stopped because:

- internal validation performance was no longer improving
- mAP50-95 had declined from the Epoch 1 result
- training loss had increased
- completing 80 epochs would require many additional hours
- external-benchmark evidence was more valuable than continuing blindly

The training was not stopped because of a runtime crash.

---

## 10. Best MuSGD Checkpoint

The best checkpoint from the MuSGD experiment is:

```text
runs/detect/waste_v2_yolo26n/weights/best.pt
```

This checkpoint was preserved and evaluated on the frozen external benchmark.

Other checkpoints such as `last.pt` are not used as the official V2 model.

---

## 11. AdamW Fine-Tuning Experiment

A second controlled experiment was created after the MuSGD degradation.

The experiment started again from the original V1 best checkpoint, not from the degraded later V2 checkpoint.

Main parameters:

```text
Optimizer:       AdamW
lr0:             0.001
lrf:             0.1
warmup_epochs:   1.0
warmup_bias_lr:  0.001
Batch:           8
Workers:         0
Image size:      640
Maximum epochs:  20
Patience:        5
Device:          CUDA:0
AMP:             True
```

The experiment script is preserved as:

```text
scripts/train_waste_v2_adamw_experiment.py
```

The run directory was:

```text
runs/detect/waste_v2_yolo26n_adamw_ft
```

---

## 12. AdamW Internal Validation

Completed results included:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.794 | 0.653 | 0.745 | 0.583 |
| 2 | 0.762 | 0.644 | 0.722 | 0.565 |

Training loss:

```text
Epoch 1
box_loss = 0.7352
cls_loss = 1.271
l1_loss  = 0.01165

Epoch 2
box_loss = 0.7544
cls_loss = 1.333
l1_loss  = 0.01189
```

Epoch 3 started, but the validation trend had already declined.

Because the new optimizer did not improve over the MuSGD best checkpoint and each epoch required approximately 45 minutes, this experiment was stopped early.

The best AdamW checkpoint was preserved for external comparison.

---

## 13. Frozen External Benchmark

The fixed benchmark is stored locally at:

```text
data/benchmarks/external_test_v1
```

Benchmark size:

```text
Images:     2,500
Instances:  8,603
```

Class image counts:

```text
plastic:          695 images
metal:            625 images
glass:            851 images
paper_cardboard:  692 images
```

Ground-truth annotation counts:

```text
plastic:          2,394
metal:            1,881
glass:            2,499
paper_cardboard:  1,829
total:            8,603
```

The benchmark remains excluded from all training.

It is used to compare V1 and V2 under the same external conditions.

A separate Reject challenge dataset is also preserved locally at:

```text
data/benchmarks/reject_challenge_v1
```

---

## 14. V1 External Benchmark Reference

Official V1 external result:

| Metric | V1 |
|---|---:|
| Precision | 0.510 |
| Recall | 0.294 |
| mAP50 | 0.273 |
| mAP50-95 | 0.133 |

Per-class V1 result:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| plastic | 0.507 | 0.305 | 0.260 | 0.126 |
| metal | 0.589 | 0.291 | 0.300 | 0.146 |
| glass | 0.445 | 0.288 | 0.237 | 0.104 |
| paper_cardboard | 0.499 | 0.293 | 0.295 | 0.154 |

---

## 15. V2 MuSGD External Benchmark

Checkpoint:

```text
runs/detect/waste_v2_yolo26n/weights/best.pt
```

Result:

| Metric | Result |
|---|---:|
| Precision | 0.592 |
| Recall | 0.385 |
| mAP50 | 0.424 |
| mAP50-95 | 0.281 |

Per class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| plastic | 0.642 | 0.307 | 0.367 | 0.233 |
| metal | 0.432 | 0.390 | 0.362 | 0.255 |
| glass | 0.691 | 0.448 | 0.499 | 0.328 |
| paper_cardboard | 0.602 | 0.394 | 0.467 | 0.310 |

Performance:

```text
Preprocess:   0.5 ms/image
Inference:    5.1 ms/image
Postprocess:  1.3 ms/image
```

Benchmark result directory:

```text
runs/detect/runs/benchmark_v2/v2_musgd_best
```

---

## 16. V2 AdamW External Benchmark

Checkpoint:

```text
runs/detect/waste_v2_yolo26n_adamw_ft/weights/best.pt
```

Result:

| Metric | Result |
|---|---:|
| Precision | 0.581 |
| Recall | 0.366 |
| mAP50 | 0.400 |
| mAP50-95 | 0.261 |

Per class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| plastic | 0.684 | 0.247 | 0.335 | 0.216 |
| metal | 0.473 | 0.351 | 0.350 | 0.245 |
| glass | 0.565 | 0.481 | 0.481 | 0.312 |
| paper_cardboard | 0.603 | 0.384 | 0.433 | 0.273 |

Performance:

```text
Preprocess:   0.6 ms/image
Inference:    5.8 ms/image
Postprocess:  2.1 ms/image
```

Benchmark result directory:

```text
runs/detect/runs/benchmark_v2/v2_adamw_best
```

---

## 17. Final Model Comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| V1 baseline | 0.510 | 0.294 | 0.273 | 0.133 |
| V2 AdamW | 0.581 | 0.366 | 0.400 | 0.261 |
| V2 MuSGD | 0.592 | 0.385 | 0.424 | 0.281 |

The strongest currently available external-benchmark result is produced by the V2 MuSGD checkpoint.

Compared with V1:

```text
mAP50:
0.273 -> 0.424

Absolute improvement:
+0.151

Relative improvement:
approximately +55%
```

And:

```text
mAP50-95:
0.133 -> 0.281

Absolute improvement:
+0.148

Relative improvement:
approximately +111%
```

Recall also improved:

```text
0.294 -> 0.385
```

Precision improved:

```text
0.510 -> 0.592
```

The V2 model therefore provides a substantial improvement in external generalization over V1.

---

## 18. Per-Class V1 vs V2 Improvement

### Plastic

```text
V1 mAP50:     0.260
V2 mAP50:     0.367

V1 mAP50-95:  0.126
V2 mAP50-95:  0.233
```

### Metal

```text
V1 mAP50:     0.300
V2 mAP50:     0.362

V1 mAP50-95:  0.146
V2 mAP50-95:  0.255
```

### Glass

```text
V1 mAP50:     0.237
V2 mAP50:     0.499

V1 mAP50-95:  0.104
V2 mAP50-95:  0.328
```

Glass showed the largest improvement.

### Paper / Cardboard

```text
V1 mAP50:     0.295
V2 mAP50:     0.467

V1 mAP50-95:  0.154
V2 mAP50-95:  0.310
```

All four classes improved in mAP50-95 compared with V1.

---

## 19. Current V2 Baseline

The current official V2 baseline for further development is:

```text
runs/detect/waste_v2_yolo26n/weights/best.pt
```

This checkpoint must be preserved.

It serves as the reference model for:

- error analysis
- Streamlit V2 integration
- live-camera testing
- decision / Reject development
- future V2.1 comparison
- physical prototype tests

The AdamW checkpoint is retained only as an experiment and comparison candidate.

---

## 20. Important Metric Interpretation

The external benchmark result:

```text
mAP50 = 0.424
```

must not be interpreted as a simple 42.4% real-world classification success rate.

Object-detection mAP combines multiple aspects of detector performance, including:

- localization quality
- class correctness
- confidence ranking
- false positives
- missed detections
- Intersection over Union requirements

The physical prototype will later be evaluated separately in its controlled operating environment.

A future real-world test should measure:

```text
Correct Classification Rate
Wrong Classification Rate
Miss Rate
Reject Rate
Sorting Success Rate
```

A controlled prototype test can use a fixed camera, fixed lighting, fixed distance, controlled conveyor speed, and one object at a time.

---

## 21. Decision on Additional Training

No immediate Training V3 or Training V4 is approved.

The completed experiments showed that simply increasing the number of epochs does not guarantee better generalization.

The next model-improvement process is evidence-based:

```text
Current V2 baseline
        ↓
Error Analysis
        ↓
Identify concrete failure patterns
        ↓
Targeted V2.1 dataset
        ↓
Short fine-tuning only if justified
        ↓
Frozen external benchmark
        ↓
V2 vs V2.1 comparison
```

Possible failure patterns to investigate include:

```text
small-object misses
glass / plastic confusion
metal reflections
difficult backgrounds
poor lighting
partial objects
multiple-object scenes
low-confidence detections
```

A future V2.1 experiment should focus only on observed failure modes rather than adding data or epochs randomly.

---

## 22. Planned Error Analysis

The next planned analysis script is:

```text
scripts/analyze_v2_errors.py
```

Its purpose will be to inspect the frozen benchmark and organize failures into categories such as:

```text
errors/
├── missed/
├── wrong_class/
├── low_confidence/
└── good/
```

The analysis should also report:

```text
missed detections by class
false positives
wrong-class detections
low-confidence correct detections
class confusion patterns
confidence distributions
```

The results will determine whether a targeted V2.1 training run is necessary.

---

## 23. Streamlit and Inference Status

The project already contains:

```text
ai/inference.py
ai/streamlit_app.py
```

The current Streamlit application supports:

- multiple image uploads
- image selection
- YOLO inference
- annotated image display
- detected-class display
- confidence display
- detection tables
- session statistics

However, the existing inference configuration still references the V1 model path.

It must be updated to use the selected V2 baseline:

```text
runs/detect/waste_v2_yolo26n/weights/best.pt
```

After that, the next step is live-camera inference.

---

## 24. Remaining AI and Control Integration

After V2 documentation and error analysis, the planned implementation sequence is:

```text
Streamlit with V2 best.pt
        ↓
Live camera inference
        ↓
Best-detection selection
        ↓
Decision logic
        ↓
Reject logic
        ↓
PySerial communication
        ↓
Arduino control
        ↓
Sensor integration
        ↓
Conveyor control
        ↓
Robotic-arm sorting
        ↓
Target bin
```

The four supported decisions are:

```text
PLASTIC
METAL
GLASS
PAPER_CARDBOARD
```

A fifth system-level decision will be:

```text
REJECT
```

Reject remains Python/control logic and is not a fifth YOLO class.

---

## 25. Physical Prototype Goal

The current V1 physical prototype is intended to contain:

```text
Fixed camera
Conveyor belt
Python / YOLO26n detection
Decision logic
Arduino-compatible controller
Sensor
One robotic arm
Four recycling bins
Final reject path
```

The prototype is university-scale and prioritizes functional proof of concept rather than industrial throughput.

---

## 26. Git Record

Waste V2 pipeline implementation:

```text
Commit:
68e7f19

Message:
feat: build and train Waste V2 pipeline
```

Initial V2 documentation:

```text
Commit:
387f7c9

Message:
docs: document Waste V2 dataset and training pipeline
```

The AdamW experiment is preserved separately in:

```text
scripts/train_waste_v2_adamw_experiment.py
```

Generated datasets, training runs, benchmark outputs, and model weights remain local and are intentionally excluded from Git.

---

## 27. Reproducibility Notes

The stable V2 Windows training configuration used:

```text
batch=8
workers=0
imgsz=640
seed=26
deterministic=True
amp=True
cache=False
```

The frozen external benchmark must never be added to training.

The current official V2 baseline must not be overwritten by a materially different experiment.

Any future model experiment should use a new run name.

---

## 28. Final V2 Training Status

```text
Waste V2 dataset             COMPLETE
Dataset validation           COMPLETE
Sanity training              COMPLETE
MuSGD experiment             COMPLETE
AdamW experiment             COMPLETE
Frozen benchmark comparison  COMPLETE
Current V2 baseline          SELECTED
Error analysis               NEXT
Streamlit V2 integration     PENDING
Live camera                   PENDING
Decision / Reject             PENDING
PySerial / Arduino            PENDING
Physical integration          PENDING
```

The V2 training phase is therefore:

```text
COMPLETE
```

The selected current baseline is:

```text
runs/detect/waste_v2_yolo26n/weights/best.pt
```

The model may still receive a targeted V2.1 improvement later if error analysis provides clear evidence that retraining is justified.