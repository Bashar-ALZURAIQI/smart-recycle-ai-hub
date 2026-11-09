#!/usr/bin/env python3
"""Read-only audit of local Ultralytics YOLO training outputs.
Run from a Smart Recycle AI-Hub checkout:
    python inspect_training_runs.py
No training is started and model files are not modified.
"""
import csv
import datetime as dt
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUTPUT = RUNS / "training_audit"
METRICS = {
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50"),
    "mAP50-95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    "box_loss": ("train/box_loss",),
    "cls_loss": ("train/cls_loss",),
}

def number(x):
    try:
        n = float(str(x).strip())
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None

def metric(row, name):
    for col in METRICS[name]:
        if col in row:
            return number(row[col])
    return None

def fmt(n, decimals=4):
    return f"{n:.{decimals}f}" if n is not None else "N/A"

def read_csv(path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [
                {str(k).strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in row.items() if k is not None}
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as e:
        print(f"WARNING: Cannot read {path}: {e}")
        return []

def args_info(path):
    if not path.is_file():
        return {}
    try:
        import yaml  # Included in this project's requirements
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except (ImportError, Exception):
        data = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line and not line[0].isspace() and ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()
        except OSError:
            pass
        return data

def relative(p):
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)

def stamp(ts):
    return dt.datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

def inventory():
    if not RUNS.is_dir():
        return []
    folders = set()
    for f in RUNS.rglob("results.csv"):
        folders.add(f.parent)
    for f in RUNS.rglob("best.pt"):
        if f.parent.name.lower() == "weights":
            folders.add(f.parent.parent)
    for f in RUNS.rglob("last.pt"):
        if f.parent.name.lower() == "weights":
            folders.add(f.parent.parent)
    items = []
    for folder in folders:
        results_file = folder / "results.csv"
        rows = read_csv(results_file) if results_file.is_file() else []
        best_candidates = [
            row for row in rows if metric(row, "mAP50-95") is not None
        ]
        highest = max(best_candidates, key=lambda r: metric(r, "mAP50-95")) if best_candidates else None
        last = rows[-1] if rows else None
        bestpt = folder / "weights" / "best.pt"
        lastpt = folder / "weights" / "last.pt"
        files = [f for f in (results_file, bestpt, lastpt, folder / "args.yaml") if f.is_file()]
        if not files:
            continue
        args = args_info(folder / "args.yaml")
        times = [f.stat().st_mtime for f in files]
        epoch = highest.get("epoch", "?") if highest else "?"
        items.append({
            "folder": folder,
            "path": relative(folder),
            "updated": max(times),
            "best": highest,
            "last": last,
            "epoch": epoch,
            "n_epochs": len(rows),
            "bestpt": bestpt,
            "lastpt": lastpt,
            "results_file": results_file,
            "args": args,
        })
    return sorted(items, key=lambda x: x["updated"], reverse=True)

def is_v21(s):
    return bool(re.search(r"(?i)(?:v2[._-]1|v21)(?:\b|_|-)", s))

def report(items):
    lines = [
        "# Local YOLO Training Audit",
        "",
        "**Generated:** " + stamp(dt.datetime.now().timestamp()),
        "",
        "Read-only inventory: these are local files, not proof of an external benchmark.",
        "File modification time can change when runs are copied; verify `args.yaml` and training logs.",
        "",
        "| Rank | Run | Latest file modified | Epochs in CSV | Highest val mAP50-95 | best.pt |",
        "|---:|---|---|---:|---:|---|",
    ]
    for i, item in enumerate(items, 1):
        lines.append(
            f"| {i} | `{item['path']}` | {stamp(item['updated'])} | "
            f"{item['n_epochs']} | {fmt(metric(item['best'], 'mAP50-95')) if item['best'] else 'N/A'} | "
            f"{'YES' if item['bestpt'].is_file() else 'NO'} |"
        )
    for i, item in enumerate(items, 1):
        lines.extend(["", f"## Run {i}: `{item['path']}`", "",
                      f"- Last file modification: {stamp(item['updated'])}",
                      f"- results.csv: {'YES' if item['results_file'].is_file() else 'NO'}",
                      f"- Number of logged epochs: {item['n_epochs']}",
                      f"- best.pt: {relative(item['bestpt']) if item['bestpt'].is_file() else 'MISSING'}",
                      f"- last.pt: {relative(item['lastpt']) if item['lastpt'].is_file() else 'MISSING'}"])
        a = item["args"]
        for k in ("model","data","epochs","imgsz","batch","optimizer","lr0","lrf","patience",
                  "seed","device","workers","project","name"):
            if k in a:
                lines.append(f"- {k}: `{str(a[k]).replace('`', '')}`")
        for label, row in (("Highest logged val mAP50-95 epoch", item["best"]),
                           ("Last logged epoch", item["last"])):
            if row:
                lines.extend(["", f"**{label}** (epoch `{row.get('epoch', '?')}`)",
                              "",
                              "| Metric | Value |", "|---|---:|"])
                for k in METRICS:
                    lines.append(f"| {k} | {fmt(metric(row, k))} |")
        lines.extend(["", "Note: the epoch with the highest logged validation mAP50-95 "
                       "is not an independent external-test result. Re-evaluate saved "
                       "`best.pt` against the same frozen dataset as V2.", ""])
    v2 = next((x for x in items if x["folder"].name == "waste_v2_yolo26n"), None)
    v21 = next((x for x in items if is_v21(x["path"])), None)
    lines.extend(["", "## V2 vs V2.1 (internal validation only)", ""])
    if v2 and v21 and v2["best"] and v21["best"]:
        lines.extend([f"- V2 candidate: `{v2['path']}`",
                      f"- V2.1 candidate: `{v21['path']}`", "",
                      "| Metric | V2 | V2.1 | Difference |",
                      "|---|---:|---:|---:|"])
        for k in ("precision","recall","mAP50","mAP50-95"):
            a, b = metric(v2["best"], k), metric(v21["best"], k)
            delta = (b-a) if a is not None and b is not None else None
            lines.append(f"| {k} | {fmt(a)} | {fmt(b)} | {f'{delta:+.4f}' if delta is not None else 'N/A'} |")
        lines.extend(["", "**Caution:** compare these only if both runs used the same "
                      "validation set and evaluation protocol. An external benchmark "
                      "is required for a valid model-selection claim."])
    else:
        lines.append("Automatic comparison unavailable: a distinct V2.1 run and official V2 "
                     "run were not both identified by folder name with results.csv. "
                     "Inspect the inventory and rename no files until the runs are confirmed.")
    lines.extend(["", "## What to document next", "",
                  "- Model starting weights, run name, optimizer, learning rate, seed and dataset version",
                  "- Number of completed epochs; why training stopped",
                  "- Training curves (`results.png`) and `results.csv`",
                  "- Per-class precision, recall and mAP on the **same** frozen benchmark",
                  "- Confusion matrix and examples of errors (glass/plastic, small items, reflections)",
                  "- False positives, missed objects and Reject handling",
                  "- Comparison with the unchanged V2 baseline, and limitations",
                  "- Never copy the frozen benchmark into the training set",
                  "- Do not commit model weights or large raw datasets unless explicitly intended",
                  ""])
    return "\n".join(lines)

def main():
    print(f"Project root: {ROOT}")
    print(f"Searching for local Ultralytics outputs under: {RUNS}")
    items = inventory()
    if not items:
        print("No local runs containing results.csv or weights/best.pt were found.")
        print("Make sure this file is in your project ROOT and inspect where training was saved.")
        return
    OUTPUT.mkdir(exist_ok=True)
    md = OUTPUT / "training_audit.md"
    md.write_text(report(items), encoding="utf-8")
    table = OUTPUT / "training_runs.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run","last_modified","epochs_logged","best_epoch",
                    "precision","recall","mAP50","mAP50-95","best_pt_exists"])
        for item in items:
            best = item["best"]
            w.writerow([
                item["path"], stamp(item["updated"]), item["n_epochs"], item["epoch"],
                *[fmt(metric(best, k)) if best else "" for k in ("precision","recall","mAP50","mAP50-95")],
                item["bestpt"].is_file()
            ])
    print("\nRUNS ORDERED BY RECENT LOCAL FILE MODIFICATION\n")
    for i, item in enumerate(items, 1):
        best = item["best"]
        print(f"{i}. {item['path']}")
        print(f"   Modified: {stamp(item['updated'])} | logged epochs: {item['n_epochs']}")
        print(f"   Best logged val: P={fmt(metric(best,'precision')) if best else 'N/A'}"
              f" R={fmt(metric(best,'recall')) if best else 'N/A'}"
              f" mAP50={fmt(metric(best,'mAP50')) if best else 'N/A'}"
              f" mAP50-95={fmt(metric(best,'mAP50-95')) if best else 'N/A'}"
              f" | best.pt={'YES' if item['bestpt'].is_file() else 'NO'}")
    print(f"\nReport: {md}\nTable:  {table}")
    print("No model was trained, overwritten, deleted, or moved.")

if __name__ == "__main__":
    main()
