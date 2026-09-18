\# YOLO26n V2 Training



\## 1. Purpose



This document records the official YOLO26n Waste V2 fine-tuning process for the Smart Recycle AI-Hub prototype.



The model detects exactly four waste classes:



```text

plastic

metal

glass

paper\_cardboard

```



Reject is not a YOLO class.



Unknown, unsupported, ambiguous, or low-confidence objects will be handled later by Python decision logic.



\---



\# 2. Training Strategy



V2 is not trained from random initialization.



It is fine-tuned from the best official V1 checkpoint:



```text

runs/detect/runs/waste\_v1/yolo26n\_v1\_baseline/weights/best.pt

```



This allows V2 to retain the useful knowledge learned during V1 while adapting to the larger and more diverse V2 dataset.



\---



\# 3. Dataset



Dataset configuration:



```text

data/processed/waste\_v2/data.yaml

```



Train:



```text

Images:       28,423

Annotations: 101,939

Backgrounds:  1,855

```



Validation:



```text

Images:       3,869

Annotations: 14,128

Backgrounds:   253

```



Classes:



```text

0 plastic

1 metal

2 glass

3 paper\_cardboard

```



\---



\# 4. Training Environment



Official V2 environment:



```text

Python:      3.14.6

Ultralytics: 8.4.142

PyTorch:     2.14.0+cu126

CUDA:        Enabled

GPU:         NVIDIA GeForce RTX 4050 Laptop GPU

VRAM:        approximately 6 GB

```



CUDA verification:



```text

CUDA available: True

GPU: NVIDIA GeForce RTX 4050 Laptop GPU

```



Automatic Mixed Precision checks passed.



\---



\# 5. Training Script



Official training script:



```text

scripts/train\_waste\_v2.py

```



Supported modes:



```text

sanity

full

```



The script performs a preflight validation before training.



It verifies:



\* Train image/label count,

\* Validation image/label count,

\* empty-label count,

\* annotation count,

\* class distribution,

\* valid class IDs,

\* CUDA availability,

\* GPU name,

\* GPU memory,

\* starting weight existence.



\---



\# 6. Initial Data YAML Path Failure



The first sanity attempt failed before training because Ultralytics resolved:



```yaml

path: .

```



to the repository root.



Ultralytics therefore searched for:



```text

D:\\Projects\\Smart\_Recycle\_AI-Hub\\valid\\images

```



instead of the V2 validation directory.



The dataset itself was valid.



The V2 `data.yaml` dataset root was corrected, after which Ultralytics resolved:



```text

Train -> data/processed/waste\_v2/train/images

Valid -> data/processed/waste\_v2/valid/images

```



correctly.



\---



\# 7. First Sanity Attempt



Initial sanity configuration used:



```text

batch:   automatic

workers: 4

```



Ultralytics AutoBatch selected:



```text

batch size: 9

```



The training epoch itself completed.



However, final validation failed inside a DataLoader worker with an OpenCV memory allocation error:



```text

cv2.error

OutOfMemoryError

Failed to allocate memory

```



The failure occurred in a CPU DataLoader/OpenCV worker.



It was not a CUDA out-of-memory failure.



\---



\# 8. Memory-Stability Fix



For stable Windows training, the configuration was changed to:



```text

batch: 8

workers: 0

```



Reasons:



\* batch 8 is slightly below the automatically selected batch 9,

\* workers 0 prevents multiple OpenCV/DataLoader worker processes from increasing RAM pressure,

\* the configuration remained comfortably within the RTX 4050 6 GB VRAM limit.



No dataset changes were needed.



\---



\# 9. Successful Sanity Training



Sanity configuration:



```text

Starting weights:

V1 yolo26n\_v1\_baseline best.pt



Epochs:        1

Train fraction: 0.05

Image size:    640

Batch:         8

Workers:       0

Device:        0

AMP:           True

```



Ultralytics used:



```text

1,421 training images

3,869 validation images

```



The full Validation split was retained even though only 5% of Train was used.



Sanity training completed successfully.



Validation result:



| Metric    | Result |

| --------- | -----: |

| Precision |  0.223 |

| Recall    |  0.367 |

| mAP50     |  0.197 |

| mAP50-95  |  0.105 |



Per-class result:



| Class           | Precision | Recall | mAP50 | mAP50-95 |

| --------------- | --------: | -----: | ----: | -------: |

| plastic         |     0.213 |  0.364 | 0.186 |   0.0929 |

| metal           |     0.133 |  0.542 | 0.175 |   0.0919 |

| glass           |     0.191 |  0.166 | 0.118 |   0.0629 |

| paper\_cardboard |     0.356 |  0.395 | 0.309 |    0.171 |



These metrics are not intended as final V2 performance results.



The sanity run used only one epoch and only 5% of Train.



Its purpose was to verify the complete training pipeline.



Sanity result:



```text

PASS

```



\---



\# 10. Official Full V2 Training



The official run starts again from the original V1 best checkpoint.



The sanity checkpoint is not used as the starting model.



Command:



```cmd

python scripts\\train\_waste\_v2.py --weights "runs\\detect\\runs\\waste\_v1\\yolo26n\_v1\_baseline\\weights\\best.pt" --mode full

```



Configuration:



| Parameter      | Value      |

| -------------- | ---------- |

| Starting model | V1 best.pt |

| Epochs         | 80 maximum |

| Patience       | 20         |

| Image size     | 640        |

| Batch          | 8          |

| Workers        | 0          |

| Device         | CUDA:0     |

| AMP            | True       |

| Seed           | 26         |

| Deterministic  | True       |

| Cache          | False      |

| Validation     | Enabled    |

| Save period    | 10 epochs  |



Optimizer was automatically selected by Ultralytics.



For the full V2 run, Ultralytics selected:



```text

MuSGD

lr = 0.01

momentum = 0.9

```



\---



\# 11. Model Architecture



Model:



```text

YOLO26n

```



Training summary:



```text

260 layers

2,505,360 parameters

2,505,360 gradients

5.9 GFLOPs

```



All compatible weights transferred successfully:



```text

Transferred 708/708 items from pretrained weights

```



\---



\# 12. Full Training — Epoch 1



The first official V2 epoch completed successfully.



Training loss:



```text

box\_loss: 0.6975

cls\_loss: 1.162

l1\_loss:  0.01094

```



Validation:



| Metric    | Epoch 1 |

| --------- | ------: |

| Precision |   0.809 |

| Recall    |   0.658 |

| mAP50     |   0.756 |

| mAP50-95  |   0.601 |



Epoch 1 training time was approximately:



```text

24 minutes 48 seconds

```



Validation required approximately:



```text

1 minute 6 seconds

```



These are \*\*internal V2 Validation metrics\*\*.



They must not be directly treated as final external-benchmark performance.



\---



\# 13. Current Training Status



As of 18 September 2026:



```text

Full V2 training: IN PROGRESS

Target: up to 80 epochs

Early stopping patience: 20

```



At the time this document was created, Epoch 2 had already started successfully.



The training process should not be interrupted unless a genuine runtime error occurs.



\---



\# 14. Output Directory



Official run:



```text

runs/detect/waste\_v2\_yolo26n/

```



Expected final checkpoints:



```text

runs/detect/waste\_v2\_yolo26n/weights/best.pt

runs/detect/waste\_v2\_yolo26n/weights/last.pt

```



Checkpoints and large generated training artifacts are intentionally not committed to Git.



\---



\# 15. Model Selection



The final V2 model must not be selected only using the V2 Validation split.



After training completes:



1\. identify the best V2 checkpoint,

2\. evaluate it against the frozen external benchmark,

3\. compare it with the official V1 baseline,

4\. inspect per-class behavior,

5\. evaluate Reject behavior separately,

6\. freeze the chosen model for the prototype.



The frozen benchmark must remain excluded from training.



\---



\# 16. V1 External Benchmark Reference



The previous V1 benchmark result was:



```text

Precision:   0.510

Recall:      0.294

mAP50:       0.273

mAP50-95:    0.133

```



This provides the reference point for the later V2 external-benchmark comparison.



The V2 Epoch 1 internal Validation result must not be compared as though both numbers were measured on the same dataset.



\---



\# 17. Remaining Training Workflow



Current sequence:



```text

Full V2 training

&#x20;       ↓

best.pt selection

&#x20;       ↓

Frozen external benchmark

&#x20;       ↓

V1 vs V2 comparison

&#x20;       ↓

Model freeze

&#x20;       ↓

Streamlit / inference integration

&#x20;       ↓

Live camera

&#x20;       ↓

Decision / Reject logic

&#x20;       ↓

PySerial / Arduino integration

```



\---



\# 18. Git Record



Initial V2 pipeline implementation:



```text

Commit:

68e7f19



Message:

feat: build and train Waste V2 pipeline

```



Branch:



```text

feature/waste-v2

```



This commit contains:



```text

scripts/build\_waste\_v2.py

scripts/test\_build\_waste\_v2.py

scripts/test\_roboflow\_path\_resolution.py

scripts/train\_waste\_v2.py

```



\---



\# 19. Important Reproducibility Notes



The official V2 training run uses:



```text

batch=8

workers=0

imgsz=640

seed=26

deterministic=True

```



These settings should be retained when reproducing the official run unless a new experiment is intentionally created.



Do not overwrite the official V2 run with materially different hyperparameters and still call it the same experiment.



\---



\# 20. Final Results



Status:



```text

NOT YET FINAL

```



This section must be updated after the official training run finishes.



The final record should include:



```text

Best epoch

Final Precision

Final Recall

Final mAP50

Final mAP50-95

Per-class metrics

best.pt path

Frozen benchmark result

V1 vs V2 comparison

Final model decision

```



Until those values are available, this document must continue to describe V2 training as:



```text

IN PROGRESS

```



