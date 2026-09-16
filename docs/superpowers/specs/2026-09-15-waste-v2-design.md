# Waste V2 Dataset and Training Design

Date: 2026-09-15

## 1. Purpose

Waste V2 is the second training generation of the Smart Recycle AI-Hub object-detection model.

The purpose of V2 is to improve real-world generalization after YOLO26n V1 showed good inference speed but weak performance on the frozen external benchmark.

V1 remains preserved as the baseline.

V2 must improve the detector without contaminating the external benchmark or changing the final four-class architecture.

---

## 2. Current Baseline

The current completed baseline is:

- Dataset V1 prepared

- YOLO26n V1 trained for 80/80 epochs

- `best.pt` selected

- Streamlit image inference implemented

- External benchmark builder implemented

- External benchmark executed

- Benchmark frozen from future training

V1 external benchmark:

- Images: 2500

- Instances: 8603

- Precision: 0.510

- Recall: 0.294

- mAP50: 0.273

- mAP50-95: 0.133

Inference speed was already strong.

The primary V1 problem is generalization and recall, not inference speed.

---

## 3. Final Detection Classes

The detector continues to use exactly four classes:

0 = plastic

1 = metal

2 = glass

3 = paper_cardboard

These class IDs must remain unchanged throughout V2.

`Reject` is NOT a fifth YOLO class.

Reject will be implemented later in the decision layer using confidence, detection state, and controlled negative behavior.

---

## 4. Frozen Benchmark

The following datasets are permanently excluded from training:

- `data/benchmarks/external_test_v1`

- `data/benchmarks/reject_challenge_v1`

External core benchmark:

- 2500 images

- 8603 target annotations

Reject challenge:

- 300 images

These datasets are development benchmarks only.

They must never be copied into:

- V2 train

- V2 validation

- future fine-tuning datasets

They may only be used for model evaluation.

Because the benchmark results have already been inspected, a separate unseen dataset must later be reserved for final blind acceptance testing.

---

## 5. Candidate V2 Data Sources

### 5.1 Existing Waste V1 Data

Path:

`data/processed/waste_v1`

Observed size:

- 16416 images

This provides the original V1 learning base.

Existing V1 validation examples must remain validation-side data and must not be silently moved into V2 training.

---

### 5.2 Remaining TACO Candidates

Path:

`data/raw/taco_optional`

TACO contains:

- 1500 images

- 4784 annotations

- 60 original categories

Only eligible target-category images that are NOT part of the frozen benchmark may be considered.

All benchmark matches must be excluded before V2 creation.

---

### 5.3 Remaining Garbage Classification Candidates

Path:

`data/raw/external_dataset_2/GARBAGE CLASSIFICATION`

Original classes:

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

BIODEGRADABLE is not a target detection class.

Images already used by the frozen benchmark must be excluded.

---

### 5.4 Recyclable Waste Detection

Path:

`data/raw/recyclable_waste_detection`

Observed:

- approximately 6396 images

Original classes:

- Cardboard

- Glass

- Metal

- Paper

- Plastic

Mapping:

Cardboard -> paper_cardboard

Glass -> glass

Metal -> metal

Paper -> paper_cardboard

Plastic -> plastic

This dataset is a primary new V2 training source.

---

### 5.5 Conveyor Waste Belt

Path:

`data/raw/conveyor_waste_belt`

Observed:

- approximately 1798 images

Original classes:

- Glass

- Metal-Other

- Organic Food

- Paper-Cardboard

- Plastic

Mapping:

Glass -> glass

Metal-Other -> metal

Paper-Cardboard -> paper_cardboard

Plastic -> plastic

Organic Food must NOT become a fifth YOLO class.

Clean organic-only images may be used as controlled negative/background training examples.

This source is especially valuable because its visual domain is closer to the final conveyor-belt prototype.

---

## 6. Raw Data Protection

Raw datasets must never be modified in place.

The builder reads from:

`data/raw/...`

and creates a completely new processed dataset:

`data/processed/waste_v2`

This ensures that source datasets remain reproducible and recoverable.

---

## 7. Benchmark Leakage Protection

Before any image is accepted into V2, the builder must compare it against the frozen benchmark.

Leakage protection must include:

1\. exact content hashing

2\. perceptual image hashing

3\. known source/provenance matching where available

4\. Roboflow augmentation-family grouping

Any image identified as a benchmark duplicate must be excluded from V2.

The builder must report how many candidate images were removed because of benchmark overlap.

---

## 8. Duplicate Protection

V2 must reduce duplicate and near-equivalent training samples.

The builder must:

- detect exact duplicate image content

- detect perceptual duplicate matches

- remove duplicate annotation lines

- prevent identical candidate images from being added multiple times

- keep related Roboflow augmentation siblings in the same dataset split

Example Roboflow variants belonging to the same original image must never be divided between train and validation.

This prevents artificial validation performance caused by leakage.

---

## 9. Annotation Validation

Every accepted YOLO annotation must be validated before dataset creation.

Valid labels must:

- use class IDs 0 through 3 only

- contain five YOLO values

- have normalized coordinates

- have positive width and height

- remain inside valid normalized bounds

- contain no duplicated annotation lines

Unreadable images or invalid labels must be excluded and reported.

---

## 10. Ambiguous Images

Images containing annotated objects that cannot be safely mapped to one of the four target classes must not be converted into false background examples.

If an image contains ignored annotated objects mixed with target objects and the ignored objects would create misleading unlabeled regions, the image should be excluded unless it can be mapped safely.

Data quality is more important than maximizing raw image count.

---

## 11. Controlled Negative Images

Controlled negative images may be included to improve Reject behavior.

Examples include clean Organic Food images containing no target-class objects.

These images use empty YOLO label files.

Negative images must remain a small portion of the training dataset and must not dominate target-class learning.

The purpose of these examples is to teach the detector that some visible objects should produce no target detection.

---

## 12. Train / Validation Strategy

V2 must use a group-aware and source-aware split.

Target split for newly accepted data:

- approximately 88% train

- approximately 12% validation

However, splitting must occur by image family/group, not blindly by individual image.

All augmented versions of the same source image must remain in the same split.

Existing V1 validation families remain validation-side data.

The final validation dataset should contain examples from multiple source datasets so that it measures broader generalization.

---

## 13. V2 Dataset Output

The processed dataset will be created at:

`data/processed/waste_v2`

Expected structure:

data/processed/waste_v2/

&#x20;   images/

&#x20;       train/

&#x20;       val/

&#x20;   labels/

&#x20;       train/

&#x20;       val/

&#x20;   data.yaml

&#x20;   manifest.csv

&#x20;   build_report.json

`manifest.csv` must preserve provenance information where available.

Useful manifest information includes:

- output image

- source dataset

- original source path

- original split

- image hash

- perceptual hash

- family/group ID

- final split

- negative/background flag

`build_report.json` must summarize the completed V2 build.

---

## 14. Required Build Report

Before training begins, the V2 builder must report at minimum:

- total candidate images

- total accepted images

- train image count

- validation image count

- images per source

- images containing each final class

- annotations per class

- negative/background image count

- invalid files removed

- exact duplicates removed

- perceptual duplicates removed

- benchmark-overlap images removed

- train/validation family leakage check

Training must not start if benchmark leakage is detected.

---

## 15. V2 Model Strategy

The first V2 model will remain YOLO26n.

The first V2 training run will fine-tune from the existing V1 `best.pt` rather than restarting blindly from zero.

Reason:

- V1 already learned useful waste features

- YOLO26n is fast enough for the final prototype

- V2 should first test whether stronger and more diverse data solves the generalization problem

After V2 training:

1\. evaluate on V2 validation

2\. evaluate on the frozen 2500-image external benchmark

3\. compare directly with V1

4\. inspect per-class behavior

5\. inspect Reject challenge behavior

If YOLO26n V2 remains insufficient, the next controlled escalation is YOLO26s using the same V2 dataset.

We must not repeatedly change both model architecture and dataset at the same time without measuring the cause of improvement.

---

## 16. Development Quality Target

The final controlled conveyor system aims for approximately:

- Precision >= 0.80

- Recall >= 0.80

These are development targets, not guaranteed benchmark results.

The frozen external benchmark is expected to improve substantially over V1, but final acceptance will depend on performance in the actual controlled conveyor environment as well as a later blind test.

---

## 17. Real Conveyor Domain Fine-Tuning

After the base V2 model is working, future fine-tuning should include images captured using the project's actual:

- camera

- conveyor

- lighting

- object distance

- background

- pickup area

This domain-specific dataset will be much smaller than the general V2 dataset but highly valuable for the final prototype.

It must not replace general training data.

---

## 18. Fast Execution Strategy

The project schedule is time-sensitive.

Therefore V2 work should follow this sequence:

1\. audit all candidate datasets

2\. create and test the V2 builder

3\. build `waste_v2`

4\. validate leakage, labels, classes, and split

5\. start YOLO26n V2 training

6\. while the GPU is training, continue non-training project planning and prepare the live-camera stage

7\. evaluate V2 immediately after training

8\. continue to live camera and decision logic as soon as model quality is acceptable

Training time should be used productively rather than waiting for the GPU.

---

## 19. Post-V2 Prototype Pipeline

Once the model quality gate is passed, development continues in this order:

YOLO model

-> Live Camera

-> Decision / Reject Logic

-> PySerial

-> Arduino

-> Conveyor Control

-> Robotic Arm

-> Sensors

-> Integrated Prototype

-> Reliability Testing

-> Demo Preparation

---

## 20. Schedule Protection

The target dates remain:

- Integrated prototype target: 21 September 2026

- Final acceptance/demo readiness target: 25 September 2026

V2 must therefore be executed quickly and methodically.

The project must avoid unnecessary model experimentation once a sufficiently reliable detector is available.

After 21 September, priority changes to:

- reliability

- bug fixing

- integration stability

- repeatable demonstrations

- documentation

- final presentation readiness

No unnecessary new features should be introduced after the integrated prototype milestone.

---

## 21. V2 Success Criteria

Waste V2 is considered successfully prepared when:

- no frozen benchmark images are present

- all labels use only classes 0-3

- no known train/validation family leakage exists

- invalid annotations are removed

- source provenance is recorded

- dataset statistics are generated

- controlled negatives are limited and documented

- V2 training can begin reproducibly

YOLO26n V2 is considered an improvement only after measured comparison against V1.

The V1 benchmark remains the baseline:

- Precision: 0.510

- Recall: 0.294

- mAP50: 0.273

- mAP50-95: 0.133

---

## 22. Final Rule

The goal is not to create the largest possible dataset.

The goal is to create the cleanest, most diverse, leakage-safe dataset that improves the Smart Recycle AI-Hub detector for the real conveyor prototype.
