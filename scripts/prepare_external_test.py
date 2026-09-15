
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

FINAL_NAMES = {
    0: "plastic",
    1: "metal",
    2: "glass",
    3: "paper_cardboard",
}

# External Dataset 2:
# 0 BIODEGRADABLE, 1 CARDBOARD, 2 GLASS, 3 METAL, 4 PAPER, 5 PLASTIC
DATASET2_MAP = {
    1: 3,
    2: 2,
    3: 1,
    4: 3,
    5: 0,
}

# TACO category IDs -> our four V1 classes.
TACO_MAP = {}
for _cid in {4,5,7,21,24,27,29,36,37,38,39,40,41,42,43,44,45,47,48,49,54,55,57}:
    TACO_MAP[_cid] = 0
for _cid in {0,8,10,11,12,28,50,52}:
    TACO_MAP[_cid] = 1
for _cid in {6,9,23,26}:
    TACO_MAP[_cid] = 2
for _cid in {13,14,15,16,17,18,19,20,30,32,33,34,56}:
    TACO_MAP[_cid] = 3

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    source: str
    image_path: Path
    label_lines: list[str]
    classes: set[int]
    unknown_labels: list[str]
    split: str = ""


def remap_dataset2_class(class_id: int):
    return DATASET2_MAP.get(class_id)


def coco_bbox_to_yolo(bbox, image_width: int, image_height: int):
    x, y, w, h = map(float, bbox)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image dimensions.")
    xc = (x + w / 2.0) / image_width
    yc = (y + h / 2.0) / image_height
    wn = w / image_width
    hn = h / image_height
    return xc, yc, wn, hn


def choose_primary_class(classes: set[int], pool_frequency: dict[int, int]) -> int:
    return min(classes, key=lambda c: (pool_frequency.get(c, 0), c))


def dhash64(path: Path) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def short_file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def all_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_training_hashes(training_root: Path):
    hashes = set()
    total = 0
    unreadable = 0
    for p in all_images(training_root):
        total += 1
        h = dhash64(p)
        if h is None:
            unreadable += 1
            continue
        hashes.add(h)
    return hashes, total, unreadable


def find_matching_image(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Roboflow normally uses matching stems; fallback for unusual casing/extensions.
    for p in images_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def load_dataset2(root: Path):
    core = []
    reject = []
    split_order = ["test", "valid", "train"]

    for split in split_order:
        labels_dir = root / split / "labels"
        images_dir = root / split / "images"
        if not labels_dir.exists():
            continue

        for label_path in labels_dir.glob("*.txt"):
            image_path = find_matching_image(images_dir, label_path.stem)
            if image_path is None:
                continue

            out_lines = []
            mapped_classes = set()
            unknown = []
            has_biodegradable = False

            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                old_id = int(float(parts[0]))
                if old_id == 0:
                    has_biodegradable = True
                    unknown.append("BIODEGRADABLE")
                    continue

                new_id = remap_dataset2_class(old_id)
                if new_id is None:
                    unknown.append(f"class_{old_id}")
                    continue

                mapped_classes.add(new_id)
                out_lines.append(" ".join([str(new_id)] + parts[1:5]))

            # Core detection set must be clean: at least one target object and no
            # biodegradable/unknown annotations. This avoids treating unlabelled
            # unknown objects as background during mAP evaluation.
            if mapped_classes and not has_biodegradable and not unknown:
                core.append(Candidate(
                    source="dataset2",
                    image_path=image_path,
                    label_lines=out_lines,
                    classes=mapped_classes,
                    unknown_labels=[],
                    split=split,
                ))
            elif not mapped_classes and (has_biodegradable or unknown):
                reject.append(Candidate(
                    source="dataset2_reject",
                    image_path=image_path,
                    label_lines=[],
                    classes=set(),
                    unknown_labels=unknown or ["BIODEGRADABLE"],
                    split=split,
                ))

    return core, reject


def locate_taco_image(source_root: Path, json_parent: Path, file_name: str, basename_index):
    candidates = [
        json_parent / file_name,
        source_root / file_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return basename_index.get(Path(file_name).name)


def load_taco(source_root: Path, annotations_path: Path):
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    category_names = {c["id"]: c["name"] for c in data.get("categories", [])}
    basename_index = {}
    for p in all_images(source_root):
        basename_index.setdefault(p.name, p)

    core = []
    reject = []

    for image in data.get("images", []):
        image_id = image["id"]
        anns = anns_by_image.get(image_id, [])
        if not anns:
            continue

        image_path = locate_taco_image(
            source_root,
            annotations_path.parent,
            image["file_name"],
            basename_index,
        )
        if image_path is None or not image_path.exists():
            continue

        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            height, width = img.shape[:2]

        out_lines = []
        mapped_classes = set()
        unknown = []

        for ann in anns:
            cid = int(ann["category_id"])
            new_id = TACO_MAP.get(cid)
            if new_id is None:
                unknown.append(category_names.get(cid, f"class_{cid}"))
                continue

            xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], width, height)
            # Clip tiny numerical overflow, and skip invalid boxes.
            vals = [max(0.0, min(1.0, v)) for v in (xc, yc, wn, hn)]
            if vals[2] <= 0 or vals[3] <= 0:
                continue
            mapped_classes.add(new_id)
            out_lines.append(
                f"{new_id} {vals[0]:.8f} {vals[1]:.8f} {vals[2]:.8f} {vals[3]:.8f}"
            )

        # Same fairness rule as Dataset 2: core mAP uses only images whose
        # annotated objects all belong to our four target classes.
        if mapped_classes and not unknown:
            core.append(Candidate(
                source="taco",
                image_path=image_path,
                label_lines=out_lines,
                classes=mapped_classes,
                unknown_labels=[],
                split="",
            ))
        elif not mapped_classes and unknown:
            reject.append(Candidate(
                source="taco_reject",
                image_path=image_path,
                label_lines=[],
                classes=set(),
                unknown_labels=unknown,
                split="",
            ))

    return core, reject


def dedupe_candidates(candidates, forbidden_hashes):
    seen = set(forbidden_hashes)
    kept = []
    removed_training = 0
    removed_internal = 0
    unreadable = 0

    for c in candidates:
        h = dhash64(c.image_path)
        if h is None:
            unreadable += 1
            continue
        if h in forbidden_hashes:
            removed_training += 1
            continue
        if h in seen:
            removed_internal += 1
            continue
        seen.add(h)
        kept.append((c, h))

    return kept, {
        "removed_training_duplicates": removed_training,
        "removed_candidate_duplicates": removed_internal,
        "unreadable": unreadable,
    }


def balanced_select(candidates_with_hash, target_count: int, seed: int, taco_fraction: float):
    rng = random.Random(seed)
    candidates = list(candidates_with_hash)
    rng.shuffle(candidates)

    freq = Counter()
    for c, _ in candidates:
        for cls in c.classes:
            freq[cls] += 1

    buckets = defaultdict(list)
    for c, h in candidates:
        primary = choose_primary_class(c.classes, freq)
        buckets[primary].append((c, h))

    base = target_count // 4
    quotas = {i: base for i in range(4)}
    for i in range(target_count % 4):
        quotas[i] += 1

    selected = []
    used_paths = set()

    for cls in range(4):
        bucket = buckets[cls]
        taco = [x for x in bucket if x[0].source == "taco"]
        other = [x for x in bucket if x[0].source != "taco"]
        rng.shuffle(taco)
        rng.shuffle(other)

        need = quotas[cls]
        taco_need = min(len(taco), int(round(need * taco_fraction)))
        picks = taco[:taco_need]
        remainder = need - len(picks)
        picks += other[:remainder]
        remainder = need - len(picks)
        if remainder > 0:
            picks += taco[taco_need:taco_need + remainder]

        for item in picks:
            key = str(item[0].image_path.resolve())
            if key not in used_paths:
                selected.append(item)
                used_paths.add(key)

    # Fill any shortfall from the remaining clean candidates.
    if len(selected) < target_count:
        for item in candidates:
            key = str(item[0].image_path.resolve())
            if key in used_paths:
                continue
            selected.append(item)
            used_paths.add(key)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


def copy_core_set(selected, out_root: Path):
    images_out = out_root / "images"
    labels_out = out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    class_image_counts = Counter()
    source_counts = Counter()
    annotation_counts = Counter()

    for idx, (c, dh) in enumerate(selected, start=1):
        suffix = c.image_path.suffix.lower()
        if suffix not in IMAGE_EXTS:
            suffix = ".jpg"
        unique = short_file_hash(c.image_path)
        stem = f"{c.source}_{idx:04d}_{unique}"
        out_image = images_out / f"{stem}{suffix}"
        out_label = labels_out / f"{stem}.txt"

        shutil.copy2(c.image_path, out_image)
        out_label.write_text("\n".join(c.label_lines) + "\n", encoding="utf-8")

        source_counts[c.source] += 1
        for cls in c.classes:
            class_image_counts[cls] += 1
        for line in c.label_lines:
            annotation_counts[int(line.split()[0])] += 1

        manifest_rows.append({
            "new_file": out_image.name,
            "source": c.source,
            "source_split": c.split,
            "original_path": str(c.image_path),
            "classes": ",".join(FINAL_NAMES[x] for x in sorted(c.classes)),
            "dhash64": f"{dh:016x}",
        })

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys() if manifest_rows else [
            "new_file", "source", "source_split", "original_path", "classes", "dhash64"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return source_counts, class_image_counts, annotation_counts


def copy_reject_set(reject_with_hash, out_root: Path, count: int, seed: int):
    rng = random.Random(seed + 1000)
    items = list(reject_with_hash)
    rng.shuffle(items)
    items = items[:count]

    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    source_counts = Counter()

    for idx, (c, dh) in enumerate(items, start=1):
        suffix = c.image_path.suffix.lower()
        unique = short_file_hash(c.image_path)
        stem = f"{c.source}_{idx:04d}_{unique}"
        out_image = images_out / f"{stem}{suffix}"
        shutil.copy2(c.image_path, out_image)
        source_counts[c.source] += 1
        rows.append({
            "new_file": out_image.name,
            "source": c.source,
            "source_split": c.split,
            "original_path": str(c.image_path),
            "unknown_labels": "|".join(sorted(set(c.unknown_labels))),
            "dhash64": f"{dh:016x}",
        })

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
            "new_file", "source", "source_split", "original_path", "unknown_labels", "dhash64"
        ])
        writer.writeheader()
        writer.writerows(rows)

    return len(items), source_counts


def write_yaml(out_root: Path):
    p = out_root.resolve().as_posix()
    text = (
        f"path: {p}\n"
        "val: images\n"
        "test: images\n"
        "nc: 4\n"
        "names:\n"
        "  0: plastic\n"
        "  1: metal\n"
        "  2: glass\n"
        "  3: paper_cardboard\n"
    )
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build a clean, external YOLO detection test set for Smart Recycle AI-Hub."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target", type=int, default=2500)
    parser.add_argument("--reject-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taco-fraction", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    taco_root = root / "data" / "raw" / "taco_optional"
    taco_json = taco_root / "data" / "annotations.json"
    dataset2_root = root / "data" / "raw" / "external_dataset_2" / "GARBAGE CLASSIFICATION"
    training_root = root / "data" / "processed" / "waste_v1"
    out_root = root / "data" / "external_test_v1"
    reject_root = root / "data" / "reject_challenge_v1"

    required = [taco_json, dataset2_root, training_root]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required path(s):\n- " + "\n- ".join(missing))

    for out in [out_root, reject_root]:
        if out.exists():
            if not args.overwrite:
                raise SystemExit(
                    f"{out} already exists. Re-run with --overwrite if you want to rebuild it."
                )
            shutil.rmtree(out)

    print("[1/6] Hashing the original training/validation images...")
    training_hashes, train_total, train_unreadable = build_training_hashes(training_root)
    print(f"      images scanned: {train_total}, unique dHashes: {len(training_hashes)}, unreadable: {train_unreadable}")

    print("[2/6] Reading TACO...")
    taco_core, taco_reject = load_taco(taco_root, taco_json)
    print(f"      clean core candidates: {len(taco_core)}, reject-only candidates: {len(taco_reject)}")

    print("[3/6] Reading Garbage Classification...")
    d2_core, d2_reject = load_dataset2(dataset2_root)
    print(f"      clean core candidates: {len(d2_core)}, reject-only candidates: {len(d2_reject)}")

    print("[4/6] Removing perceptual duplicates against waste_v1 and within candidate pools...")
    core_with_hash, core_stats = dedupe_candidates(taco_core + d2_core, training_hashes)
    reject_with_hash, reject_stats = dedupe_candidates(taco_reject + d2_reject, training_hashes)
    print(f"      core usable after dedupe: {len(core_with_hash)}")
    print(f"      core dedupe stats: {core_stats}")

    if len(core_with_hash) < args.target:
        raise SystemExit(
            f"Only {len(core_with_hash)} clean unique core images remain, below requested target {args.target}. "
            "Use a lower --target or add another unseen external dataset."
        )

    print(f"[5/6] Selecting {args.target} images with class/source balancing...")
    selected = balanced_select(core_with_hash, args.target, args.seed, args.taco_fraction)
    source_counts, class_image_counts, annotation_counts = copy_core_set(selected, out_root)
    write_yaml(out_root)

    print(f"[6/6] Preparing up to {args.reject_count} unknown-only images for the later Reject test...")
    reject_n, reject_sources = copy_reject_set(
        reject_with_hash, reject_root, args.reject_count, args.seed
    )

    print("\nDONE")
    print(f"Core test set: {out_root}")
    print(f"Core images: {len(selected)}")
    print("Sources:", dict(source_counts))
    print("Images containing each final class:", {FINAL_NAMES[k]: class_image_counts[k] for k in range(4)})
    print("Annotations by final class:", {FINAL_NAMES[k]: annotation_counts[k] for k in range(4)})
    print(f"Reject challenge: {reject_root} ({reject_n} images)")
    print("Reject sources:", dict(reject_sources))
    print("\nNext evaluation command will use:")
    print(out_root / "data.yaml")


if __name__ == "__main__":
    main()
