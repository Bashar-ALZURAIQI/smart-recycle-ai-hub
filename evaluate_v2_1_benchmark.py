"""Read-only, like-for-like external benchmark for Smart Recycle AI-Hub.

Run from the project root with:
    python evaluate_v2_1_benchmark.py

Creates ONLY new evaluation files under runs/benchmark_v2_1_comparison/.
Never calls train(), modifies original images/weights, or updates GitHub.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'data' / 'benchmarks' / 'external_test_v1'
OUT = ROOT / 'runs' / 'benchmark_v2_1_comparison'
MODELS = {
    'V2': ROOT / 'runs' / 'detect' / 'waste_v2_yolo26n' / 'weights' / 'best.pt',
    'V2.1': ROOT / 'runs' / 'detect' / 'waste_v2_1_yolo26n' / 'weights' / 'best.pt',
}
CLASSES = ('plastic', 'metal', 'glass', 'paper_cardboard')


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value):
    return f'{value:.4f}' if isinstance(value, (float, int)) else 'N/A'


def result_record(metrics):
    box = metrics.box
    record = {
        'precision': as_float(box.mp),
        'recall': as_float(box.mr),
        'mAP50': as_float(box.map50),
        'mAP50-95': as_float(box.map),
        'speed_ms': {str(k): as_float(v) for k, v in (getattr(metrics, 'speed', None) or {}).items()},
        'per_class': {},
        'ultralytics_output': str(getattr(metrics, 'save_dir', '')),
    }
    present = list(getattr(box, 'ap_class_index', []))
    precision = getattr(box, 'p', [])
    recall = getattr(box, 'r', [])
    map50 = getattr(box, 'ap50', [])
    map5095 = getattr(box, 'ap', [])
    for i, idx in enumerate(present):
        cls = CLASSES[int(idx)]
        record['per_class'][cls] = {
            'precision': as_float(precision[i]) if i < len(precision) else None,
            'recall': as_float(recall[i]) if i < len(recall) else None,
            'mAP50': as_float(map50[i]) if i < len(map50) else None,
            'mAP50-95': as_float(map5095[i]) if i < len(map5095) else None,
        }
    return record


def write_report(data):
    lines = [
        '# Frozen External Benchmark — V2 vs V2.1', '',
        f"Generated: {data['created_at']}", '',
        f"Benchmark directory: `{data['benchmark_dir']}`", '',
        f"Number of benchmark images: {data['images']}", '',
        'Both models were evaluated on exactly the same frozen external benchmark',
        'with identical inference settings (image size 640, batch 8, GPU 0).', '',
        '| Metric | V2 | V2.1 | Delta (V2.1 − V2) |',
        '|---|---:|---:|---:|',
    ]
    v2 = data['results']['V2']
    v21 = data['results']['V2.1']
    for key in ('precision', 'recall', 'mAP50', 'mAP50-95'):
        delta = v21[key] - v2[key]
        lines.append(f'| {key} | {fmt(v2[key])} | {fmt(v21[key])} | {delta:+.4f} |')
    lines.extend(['', '## Per-class metrics', '',
                  '| Class | V2 mAP50 | V2.1 mAP50 | V2 mAP50-95 | V2.1 mAP50-95 |',
                  '|---|---:|---:|---:|---:|'])
    for cls in CLASSES:
        a = v2['per_class'].get(cls, {})
        b = v21['per_class'].get(cls, {})
        lines.append(f'| {cls} | {fmt(a.get("mAP50"))} | {fmt(b.get("mAP50"))} | '
                     f'{fmt(a.get("mAP50-95"))} | {fmt(b.get("mAP50-95"))} |')
    lines.extend(['', '## Evidence locations', ''])
    for label in MODELS:
        lines.append(f'- {label}: `{data["results"][label]["ultralytics_output"]}`')
    lines.extend(['', '## Interpretation', '',
                  'Treat this 2,500-image benchmark as a **development benchmark**, not a final blind test.',
                  'Do not select hyperparameters repeatedly based on this same benchmark.',
                  'After choosing a candidate, evaluate it once on a separate unseen test set',
                  'and measure actual physical sorting and reject-policy performance.', ''])
    return '\n'.join(lines)


def main():
    # Preflight before importing Ultralytics or writing anything.
    for label, path in MODELS.items():
        if not path.is_file():
            raise FileNotFoundError(f'{label} weights not found: {path}')
    images = BENCH / 'images'
    labels = BENCH / 'labels'
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f'Benchmark image/label folders not found at {BENCH}')
    image_paths = [p for p in images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp')]
    if not image_paths:
        raise RuntimeError(f'No benchmark images at {images}')
    print(f'Benchmark: {BENCH} | images: {len(image_paths)}')
    if len(image_paths) != 2500:
        raise RuntimeError(f'Expected 2500 frozen external images, found {len(image_paths)}. Stopping.')
    OUT.mkdir(parents=True, exist_ok=True)
    empty_train = OUT / 'empty_train' / 'images'
    empty_train.mkdir(parents=True, exist_ok=True)
    # The 'train' key is required by YOLO dataset YAML; it intentionally points
    # to an EMPTY scratch directory, NOT the protected benchmark images.
    data_yaml = OUT / 'evaluation_only.yaml'
    data_yaml.write_text(
        '# EVALUATION ONLY: Do NOT use this YAML for training.\n'
        f'path: "{BENCH.resolve().as_posix()}"\n'
        f'train: "{empty_train.resolve().as_posix()}"\n'
        'val: images\n'
        'test: images\n'
        'names:\n  0: plastic\n  1: metal\n  2: glass\n  3: paper_cardboard\n',
        encoding='utf-8',
    )
    from ultralytics import YOLO
    data = {
        'created_at': datetime.now().astimezone().isoformat(),
        'benchmark_dir': str(BENCH),
        'images': len(image_paths),
        'parameters': {'imgsz':640, 'batch':8, 'device':0, 'workers':0, 'split':'test'},
        'weights': {k: str(v) for k, v in MODELS.items()},
        'results': {},
    }
    for name, path in MODELS.items():
        print(f'\n=== Validating {name} on external benchmark (no training) ===')
        model = YOLO(str(path))
        names = model.names
        names_ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names_ordered != list(CLASSES):
            raise ValueError(f'Unexpected classes for {name}: {names_ordered}')
        metrics = model.val(
            data=str(data_yaml.resolve()),
            split='test', imgsz=640, batch=8, device=0, workers=0,
            plots=True, verbose=True, project=str(OUT.resolve()),
            name='v2_reference' if name == 'V2' else 'v2_1_candidate',
            exist_ok=False,
        )
        data['results'][name] = result_record(metrics)
        (OUT / 'partial_results.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (OUT / 'benchmark_comparison.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (OUT / 'benchmark_comparison.md').write_text(write_report(data), encoding='utf-8')
    with (OUT / 'benchmark_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model','precision','recall','mAP50','mAP50-95'])
        for name, record in data['results'].items():
            w.writerow([name] + [record[k] for k in ('precision','recall','mAP50','mAP50-95')])
    print('\nDone. No model training was performed.')
    print('Main report:', OUT / 'benchmark_comparison.md')
    print('Machine-readable results:', OUT / 'benchmark_comparison.json')


if __name__ == '__main__':
    main()
