\# Smart Recycle AI-Hub — V2 Dataset Preparation and Validation



\## Document Purpose



This document records the complete preparation process used to build the \*\*Smart Recycle AI-Hub Waste V2\*\* object-detection dataset.



It serves as:



\* a reproducibility record,

\* a technical handoff document,

\* an audit trail for source selection and filtering,

\* a record of benchmark leakage prevention,

\* a reference for future dataset revisions,

\* and the official description of the dataset used for YOLO26n V2 fine-tuning.



The raw datasets are treated as read-only.



All class remapping, filtering, deduplication, benchmark exclusion, family grouping, train/validation splitting, and negative-sample control are performed when building the processed V2 dataset.



\---



\# 1. Detection Scope



Waste V2 keeps the same four trainable classes as V1:



| Target ID | Final Class     |

| --------: | --------------- |

|         0 | plastic         |

|         1 | metal           |

|         2 | glass           |

|         3 | paper\_cardboard |



The physical prototype also contains a \*\*Reject\*\* path.



However:



> \*\*Reject is NOT a YOLO training class.\*\*



Reject will be implemented later in Python decision logic.



Possible Reject cases include:



\* unknown objects,

\* unsupported materials,

\* organic waste,

\* low-confidence detections,

\* ambiguous detections,

\* objects that do not safely map to one of the four supported classes.



\---



\# 2. Why V2 Was Created



The original V1 dataset was sufficient to establish the first YOLO26n baseline.



V2 was created to improve generalization by combining multiple independent waste datasets while preventing contamination of the frozen external benchmark.



The main V2 objectives were:



1\. retain the useful V1 training data,

2\. add new independent waste datasets,

3\. remap all sources into one four-class schema,

4\. prevent train/validation family leakage,

5\. exclude images belonging to the frozen benchmark,

6\. remove exact and perceptual duplicates,

7\. include a controlled number of negative examples,

8\. preserve source provenance,

9\. produce a reproducible YOLO dataset,

10\. prepare the dataset for YOLO26n fine-tuning.



\---



\# 3. Source Datasets



The final fast V2 build combines four sources.



\## 3.1 Waste V1



Processed source:



```text

data/processed/waste\_v1

```



Class mapping is unchanged:



```text

0 -> plastic

1 -> metal

2 -> glass

3 -> paper\_cardboard

```



Final retained V2 records:



```text

16,403

```



\---



\## 3.2 Garbage Classification



Source:



```text

data/raw/external\_dataset\_2/GARBAGE CLASSIFICATION

```



Original classes:



```text

BIODEGRADABLE

CARDBOARD

GLASS

METAL

PAPER

PLASTIC

```



Mapping:



| Original      | V2                     |

| ------------- | ---------------------- |

| BIODEGRADABLE | negative / unsupported |

| CARDBOARD     | paper\_cardboard        |

| GLASS         | glass                  |

| METAL         | metal                  |

| PAPER         | paper\_cardboard        |

| PLASTIC       | plastic                |



Final retained V2 records:



```text

7,888

```



\---



\## 3.3 Recyclable Waste Detection



Source:



```text

data/raw/recyclable\_waste\_detection

```



Mapping:



| Original  | V2              |

| --------- | --------------- |

| Cardboard | paper\_cardboard |

| Glass     | glass           |

| Metal     | metal           |

| Paper     | paper\_cardboard |

| Plastic   | plastic         |



Final retained V2 records:



```text

6,252

```



\---



\## 3.4 Conveyor Waste Belt



Source:



```text

data/raw/conveyor\_waste\_belt

```



Mapping:



| Original        | V2                     |

| --------------- | ---------------------- |

| Glass           | glass                  |

| Metal-Other     | metal                  |

| Organic Food    | negative / unsupported |

| Paper-Cardboard | paper\_cardboard        |

| Plastic         | plastic                |



Final retained V2 records:



```text

1,749

```



\---



\# 4. Optional TACO Source



TACO was evaluated as an optional source during V2 planning.



For the fast V2 build used for the prototype deadline, TACO was intentionally not included.



The build report records:



```text

optional\_taco: not included in fast V2 build

```



TACO can be reconsidered in a future V3 dataset.



\---



\# 5. Raw Dataset Policy



Raw source datasets are not modified.



The builder reads from:



```text

data/raw/

```



and writes the processed dataset into:



```text

data/processed/waste\_v2/

```



This protects the original downloads and makes the processing pipeline reproducible.



\---



\# 6. Roboflow Path Compatibility



Several downloaded Roboflow datasets contained YAML paths such as:



```yaml

train: ../train/images

val: ../valid/images

test: ../test/images

```



The extracted directory structure actually stored those directories directly under the dataset root.



The V2 builder therefore implements path-resolution fallback logic.



It first attempts the YAML path literally.



If that path does not exist, leading `.` and `..` components are removed and the builder retries relative to the actual dataset root.



A regression test was added:



```text

scripts/test\_roboflow\_path\_resolution.py

```



This prevents the three Roboflow-style sources from silently producing zero candidates.



\---



\# 7. Benchmark Protection



The V2 training dataset must not contain images from the frozen evaluation sets.



Protected benchmark directories include:



```text

data/benchmarks/external\_test\_v1

data/benchmarks/reject\_challenge\_v1

```



The builder calculates image fingerprints and removes candidate images that match protected benchmark images.



Two forms of comparison are used:



\* exact fingerprint comparison,

\* perceptual fingerprint comparison.



Final exclusions:



```text

Exact benchmark matches:      2,318

Perceptual benchmark matches:   105

```



This separation is essential because the external benchmark must remain independent of training.



\---



\# 8. Duplicate Removal



Candidate images are deduplicated before the final split.



The pipeline uses:



\* SHA-256 exact fingerprints,

\* perceptual dHash fingerprints,

\* image dimensions.



Additional perceptual duplicates removed:



```text

86

```



The V1 source is processed first, so where equivalent images are found across sources, the existing V1 copy is normally retained.



\---



\# 9. Roboflow Augmentation Families



Images that represent augmentation siblings must not be split between Train and Validation.



The pipeline assigns a `family\_id` to related images.



Family-aware splitting ensures that one family is assigned entirely to one split.



Final validation result:



```text

family\_overlap: 0

```



Therefore no detected family exists in both Train and Validation.



\---



\# 10. Ambiguous Mixed Images



Some source images contain both:



\* a supported target object,

\* and an annotated unsupported object.



Removing only the unsupported annotation would incorrectly teach YOLO that the unsupported annotated object is background.



Therefore ambiguous mixed images are skipped.



Final counts:



```text

conveyor\_waste\_belt:ambiguous\_mixed = 39

garbage\_classification:ambiguous\_mixed = 208

```



This is intentional.



\---



\# 11. Negative Samples



V2 introduces controlled negative examples.



A negative image contains an explicitly annotated unsupported source category that is converted into an empty YOLO label.



Examples include:



```text

BIODEGRADABLE

Organic Food

```



The final V2 dataset contains:



```text

Positive records: 30,184

Negative records:  2,108

```



Negative fraction:



```text

approximately 6.53%

```



The configured upper limit is:



```text

20%

```



Therefore negatives remain controlled and do not dominate training.



\---



\# 12. Final Dataset Size



Initial source candidates:



```text

34,801

```



After benchmark exclusion and deduplication:



```text

32,292

```



Final split:



```text

Train: 28,423

Valid:  3,869

Total: 32,292

```



Approximate split ratio:



```text

Train: 88.02%

Valid: 11.98%

```



\---



\# 13. Source Distribution



Final dataset:



| Source                     |    Records |

| -------------------------- | ---------: |

| waste\_v1                   |     16,403 |

| garbage\_classification     |      7,888 |

| recyclable\_waste\_detection |      6,252 |

| conveyor\_waste\_belt        |      1,749 |

| \*\*Total\*\*                  | \*\*32,292\*\* |



Train:



| Source                     | Records |

| -------------------------- | ------: |

| waste\_v1                   |  14,471 |

| garbage\_classification     |   6,929 |

| recyclable\_waste\_detection |   5,493 |

| conveyor\_waste\_belt        |   1,530 |



Validation:



| Source                     | Records |

| -------------------------- | ------: |

| waste\_v1                   |   1,932 |

| garbage\_classification     |     959 |

| recyclable\_waste\_detection |     759 |

| conveyor\_waste\_belt        |     219 |



All four sources are represented in Validation.



\---



\# 14. Annotation Distribution



\## Train



```text

Images:       28,423

Annotations: 101,939

Empty labels: 1,855

```



Class distribution:



```text

plastic:          21,930

metal:            17,178

glass:            22,612

paper\_cardboard:  40,219

```



\## Validation



```text

Images:       3,869

Annotations: 14,128

Empty labels:   253

```



Class distribution:



```text

plastic:           3,224

metal:             2,348

glass:             3,134

paper\_cardboard:   5,422

```



\---



\# 15. Build Validation



The completed build reported:



```text

train\_images:   28423

valid\_images:    3869

family\_overlap:     0

classes\_valid:   true

```



No unsupported class IDs were present in the final YOLO labels.



\---



\# 16. Automated Tests



Before the official V2 build, the complete script test suite was executed.



Result:



```text

Ran 27 tests



OK

```



Tests cover areas including:



\* YOLO parsing,

\* segmentation conversion,

\* source audits,

\* class remapping,

\* positive/negative handling,

\* family grouping,

\* benchmark fingerprints,

\* dataset output,

\* manifest generation,

\* build reporting,

\* Roboflow path resolution.



\---



\# 17. Output Files



The builder creates:



```text

data/processed/waste\_v2/

├── train/

│   ├── images/

│   └── labels/

├── valid/

│   ├── images/

│   └── labels/

├── data.yaml

├── manifest.csv

└── build\_report.json

```



`manifest.csv` records provenance and split membership.



`build\_report.json` records final counts and processing counters.



\---



\# 18. Ultralytics Dataset Path Issue



During the first V2 sanity-training attempt, Ultralytics interpreted:



```yaml

path: .

```



relative to the project working directory instead of the V2 dataset directory.



This caused Ultralytics to search for:



```text

<project-root>/valid/images

```



instead of:



```text

data/processed/waste\_v2/valid/images

```



For the current local training run, `data.yaml` was updated so that the dataset root resolves explicitly to:



```text

data/processed/waste\_v2

```



After this change, Ultralytics correctly resolved both Train and Validation paths.



This behavior should be considered when rebuilding V2 on another machine.



A future builder revision should make generated dataset-root handling portable without requiring a machine-specific absolute path.



\---



\# 19. Main Builder



Official builder:



```text

scripts/build\_waste\_v2.py

```



Primary responsibilities:



1\. discover configured sources,

2\. resolve dataset paths,

3\. parse YOLO annotations,

4\. map source classes,

5\. reject invalid or ambiguous data,

6\. assign family IDs,

7\. fingerprint images,

8\. exclude benchmark matches,

9\. remove duplicates,

10\. control negative samples,

11\. perform family-safe Train/Validation split,

12\. copy processed images and labels,

13\. generate `data.yaml`,

14\. generate provenance manifest,

15\. generate build report,

16\. validate the final dataset.



\---



\# 20. Git Record



The first implementation commit containing the V2 builder, tests, and training pipeline is:



```text

68e7f19

feat: build and train Waste V2 pipeline

```



Branch:



```text

feature/waste-v2

```



\---



\# 21. Status



As of 18 September 2026:



```text

V2 source integration       COMPLETE

Benchmark protection        COMPLETE

Deduplication               COMPLETE

Family-safe split           COMPLETE

Controlled negatives        COMPLETE

Build validation            COMPLETE

Automated tests             27/27 PASS

Sanity training             COMPLETE

Full YOLO26n V2 training    IN PROGRESS

```



The next dataset-level action is not to modify V2 while the official training run is active.



Any future dataset revision should be versioned separately rather than silently modifying the dataset used by the current training run.



