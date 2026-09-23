\# Waste V2.1 Source Audit



Date: 2026-09-23



\## Purpose



This document records the datasets audited before building Waste V2.1.



Final detector classes:



\- `0 = plastic`

\- `1 = metal`

\- `2 = glass`

\- `3 = paper\_cardboard`



`Reject` is handled by the decision layer and is not a YOLO class.



Raw datasets remain read-only.



\---



\## Source 1 — WhiteMind YOLO-Waste Detection v1



Root:



`data/raw/v2\_1\_sources/whitemind\_yolo\_waste`



Metadata:



\- Version: 1

\- License: CC BY 4.0



Original images:



\- Train: 2,340

\- Valid: 316

\- Test: 152

\- Total: 2,808



Original annotations:



\- 8,857



Classes:



\- `0 Biodegradable - Compost`

\- `1 Biodegradable | Compost`

\- `2 Glass`

\- `3 Metal`

\- `4 Mixed`

\- `5 Paper`

\- `6 Plastic`



Mapping:



\- `2 -> glass`

\- `3 -> metal`

\- `5 -> paper\_cardboard`

\- `6 -> plastic`



Excluded:



\- `0`

\- `1`

\- `4`



Policy:



\- Keep supported-only images.

\- Exclude the entire image if any unsupported class occurs.

\- Never strip unsupported boxes from a mixed image.

\- Do not convert excluded images into negatives.



Audit:



\- Supported-only images: 2,772

\- Excluded images: 36

\- Kept annotations: 8,675



Final mapped annotation counts:



\- plastic: 2,566

\- metal: 1,985

\- glass: 1,504

\- paper\_cardboard: 2,620



Quality:



\- Bad format: 0

\- Bad class IDs: 0

\- Bad coordinates: 0

\- Zero-size boxes: 0



Status:



`READY`



\---



\## Source 2 — Waste Detection Dataset 1 v3



Root:



`data/raw/v2\_1\_sources/waste\_detection\_dataset\_1`



Metadata:



\- Version: 3

\- License: CC BY 4.0



Original images:



\- Train: 15,759

\- Valid: 994

\- Test: 429

\- Total: 17,182



Original annotations:



\- 27,803



Empty label files:



\- 91



Supported mappings:



\### Plastic



\- `11 HDPE Bottles`

\- `20 PET Bottle`

\- `21 PET Cup`

\- `22 PS Plastic`

\- `25 Plastic Bag`

\- `26 Plastic Wrapper`

\- `29 plastic straw`



\### Metal



\- `0 Aluminium Foil`

\- `15 Metal Cans`

\- `16 Metal Scraps`



\### Glass



\- `1 Broken Glass`

\- `9 Glass Bottle`

\- `10 Glass Jars`



\### Paper/Cardboard



\- `4 Cardboard`

\- `14 Magazines`

\- `18 Newspaper`

\- `19 Office Paper`

\- `24 Paper`



Unsupported classes:



\- `2 Bulbs`

\- `3 Cables`

\- `5 Charger`

\- `6 Cylindrical battery`

\- `7 Earphone`

\- `8 Food Waste`

\- `12 Headphone`

\- `13 Kitchen Waste`

\- `17 Mobile Phone`

\- `23 Paint Containers`

\- `27 Pouch battery`

\- `28 facemask`



Policy:



\- Keep supported-only images.

\- Exclude supported-plus-unsupported images.

\- Exclude unsupported-only images.

\- Exclude empty labels.

\- Exclude the entire image if any line is not standard five-token YOLO detection format.

\- Never repair polygon annotations inside the raw dataset.



Supported-only audit before non-five-token filtering:



\- Images: 9,574

\- Annotations: 13,482



Mapped counts before non-five-token filtering:



\- plastic: 6,527

\- paper\_cardboard: 2,889

\- metal: 2,482

\- glass: 1,584



Annotation format:



\- Non-five-token lines: 325

\- Files affected: 205

\- Supported-only images affected: 36



The 36 affected supported-only images must be excluded completely.



Duplicate audit:



\- Images hashed: 17,182

\- Unique SHA-256 hashes: 17,170

\- Exact duplicate hashes across train/valid: 4



Consistency:



\- Images without labels: 0

\- Labels without images: 0



Status:



`READY WITH FILTERS`



\---



\## Source 3 — General Waste Data v4



Root:



`data/raw/v2\_1\_sources/general\_waste\_data`



Metadata:



\- Version: 4

\- License: CC BY 4.0



Original images:



\- Train: 14,648

\- Valid: 2,098

\- Test: 1,042

\- Total: 17,788



Original annotations:



\- 125,561



Excluded classes:



\- `0 BIODEGRADABLE`

\- `2 CLOTH`



Mapping:



\### Paper/Cardboard



\- `1 CARDBOARD`

\- `21 PAPER`

\- `22 PAPER cup`



\### Glass



\- `3 GLASS`

\- `4 GLASS bottle`

\- `5 GLASS bowl`

\- `6 GLASS broken`

\- `7 GLASS cap`

\- `8 GLASS jar`

\- `9 GLASS plate`

\- `10 GLASS water`



\### Metal



\- `11 METAL`

\- `12 METAL Jar`

\- `13 METAL cane`

\- `14 METAL cap`

\- `15 METAL knife`

\- `16 METAL paper`

\- `17 METAL plate`

\- `18 METAL scissor`

\- `19 METAL sheet`

\- `20 METAL spoon`



\### Plastic



\- `23 PLASTIC`

\- `24 PLASTIC bag`

\- `25 PLASTIC bottle`

\- `26 PLASTIC can`

\- `27 PLASTIC cane`

\- `28 PLASTIC cap`

\- `29 PLASTIC cup`

\- `30 PLASTIC plate`

\- `31 PLASTIC wrapper`



Policy:



\- Keep supported-only images.

\- Exclude the entire image if BIODEGRADABLE or CLOTH occurs.

\- Exclude empty labels.



Audit result:



\- Supported-only images: 13,897

\- Kept annotations: 46,915



Mapped counts:



\- paper\_cardboard: 14,816

\- glass: 12,798

\- plastic: 9,917

\- metal: 9,384



Quality:



\- Bad format: 0

\- Bad class IDs: 0

\- Bad coordinates: 0

\- Zero-size boxes: 0



Duplicate audit:



\- Images: 17,788

\- Unique hashes: 17,788

\- Duplicate copies: 0

\- Cross-split exact duplicates: 0



Status:



`READY`



\---



\## Existing Waste V2



Root:



`data/processed/waste\_v2`



Observed totals:



\- Final images: 32,292

\- Train: 28,423

\- Valid: 3,869

\- Positive: 30,184

\- Controlled negatives: 2,108

\- Family overlap: 0



Mapping:



\- `0 -> plastic`

\- `1 -> metal`

\- `2 -> glass`

\- `3 -> paper\_cardboard`



V2 remains unchanged.



Eligible existing controlled negatives may be retained.



\---



\## Protected Evaluation Roots



The following must never enter V2.1 train or validation:



\- `data/benchmarks/external\_test\_v1`

\- `data/benchmarks/reject\_challenge\_v1`

\- `data/benchmarks/real\_world\_v2\_1`



Protection must include:



\- Exact SHA-256 matching

\- Perceptual dHash + width + height matching



\---



\## Builder Policy Summary



Waste V2.1 builder will:



1\. Combine Waste V2 and the three audited external datasets.

2\. Apply strict class mappings.

3\. Exclude mixed unsupported images entirely.

4\. Exclude malformed/non-detection source labels according to policy.

5\. Keep only eligible V2 controlled negatives.

6\. Remove benchmark leakage.

7\. Remove exact duplicates.

8\. Remove perceptual duplicates.

9\. Keep families in one split.

10\. Rebuild train/valid with seed 26.

11\. Produce only classes 0–3.

12\. Write manifest and build report.

13\. Never modify raw datasets.

