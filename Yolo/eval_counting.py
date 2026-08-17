"""Center-matching + counting metrics. mAP is useless at 12x22 px, so score centers.

python eval_counting.py --model path/to/best.pt [--tta] [--out r.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

IMG_SIZE = 1024
MATCH_DIST_PX = 16.0


def slide_of(name: str) -> str:
    return name.split("tile")[0]


def load_gt_centers(label_path: Path, wh: tuple[float, float] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    pts = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                pts.append([float(parts[1]) * wh[0], float(parts[2]) * wh[1]])
    return np.array(pts, dtype=float).reshape(-1, 2)


def load_gt_for_images(images_dir: Path, labels_dir: Path) -> dict:
    from PIL import Image
    gts = {}
    for p in sorted(images_dir.glob("*.png")):
        # tiles come in 1024 and 640, so read the real size
        gts[p.name] = load_gt_centers(labels_dir / p.name.replace(".png", ".txt"),
                                      Image.open(p).size)
    return gts


def match_counts(pred: np.ndarray, gt: np.ndarray, thr_px: float) -> int:
    if len(pred) == 0 or len(gt) == 0:
        return 0
    dist = np.linalg.norm(pred[:, None] - gt[None, :], axis=2)
    ri, ci = linear_sum_assignment(dist)
    return int((dist[ri, ci] <= thr_px).sum())


def evaluate(preds: dict, gts: dict, conf: float, thr_px: float = MATCH_DIST_PX) -> dict:
    tp = fp = fn = 0
    pc, gc = [], []
    slide_pred, slide_gt = {}, {}
    for name, gt in gts.items():
        centers, scores = preds.get(name, (np.zeros((0, 2)), np.zeros(0)))
        keep = centers[scores >= conf] if len(centers) else centers
        m = match_counts(keep, gt, thr_px)
        tp += m
        fp += len(keep) - m
        fn += len(gt) - m
        pc.append(len(keep))
        gc.append(len(gt))
        s = slide_of(name)
        slide_pred[s] = slide_pred.get(s, 0) + len(keep)
        slide_gt[s] = slide_gt.get(s, 0) + len(gt)
    pc, gc = np.array(pc, dtype=float), np.array(gc, dtype=float)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    diff = pc - gc
    return {
        "conf": round(conf, 3),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "count_mae": round(float(np.abs(diff).mean()), 3),
        "count_rmse": round(float(np.sqrt((diff ** 2).mean())), 3),
        "count_bias": round(float(diff.mean()), 3),
        "count_pearson_r": round(float(np.corrcoef(pc, gc)[0, 1]), 4)
        if len(pc) > 1 and pc.std() > 0 and gc.std() > 0 else None,
        "total_pred": int(pc.sum()),
        "total_gt": int(gc.sum()),
        "per_slide": {s: {"pred": slide_pred[s], "gt": slide_gt[s]}
                      for s in sorted(slide_gt)},
    }


def sweep(preds: dict, gts: dict, confs=None) -> dict:
    confs = confs if confs is not None else np.round(np.arange(0.05, 0.76, 0.05), 2)
    rows = [evaluate(preds, gts, float(c)) for c in confs]
    best_mae = min(rows, key=lambda r: r["count_mae"])
    best_f1 = max(rows, key=lambda r: r["f1"])
    return {"sweep": rows, "best_count_mae": best_mae, "best_f1": best_f1}


def yolo_predict_centers(model_path: str, images_dir: Path, tta: bool = False,
                         imgsz: int = IMG_SIZE, base_conf: float = 0.01) -> dict:
    import torch
    from ultralytics import YOLO
    model = YOLO(model_path)
    preds = {}
    images = sorted(images_dir.glob("*.png"))
    bs = 8
    i = 0
    while i < len(images):
        chunk = images[i:i + bs]
        try:
            results = model.predict([str(p) for p in chunk], imgsz=imgsz,
                                    conf=base_conf, iou=0.5, max_det=1000,
                                    augment=tta, verbose=False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = max(1, bs // 2)  # someone else may be on the gpu
                continue
            raise
        for p, r in zip(chunk, results):
            xywh = r.boxes.xywh.cpu().numpy()
            preds[p.name] = (xywh[:, :2].copy(), r.boxes.conf.cpu().numpy().copy())
        i += len(chunk)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="Data/Yolo_Merged/dataset/images/test")
    ap.add_argument("--labels", default="Data/Yolo_Merged/dataset/labels/test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    images_dir, labels_dir = Path(args.images), Path(args.labels)
    gts = load_gt_for_images(images_dir, labels_dir)
    preds = yolo_predict_centers(args.model, images_dir, tta=args.tta,
                                 imgsz=args.imgsz)
    report = sweep(preds, gts)
    report["model"] = args.model
    report["tta"] = args.tta
    report["n_tiles"] = len(gts)

    b = report["best_count_mae"]
    f = report["best_f1"]
    print(f"model: {args.model}  (tta={args.tta}, {len(gts)} tiles)")
    print(f"  best F1        : {f['f1']:.3f} (P {f['precision']:.3f} / R {f['recall']:.3f}) "
          f"@ conf {f['conf']}  | count MAE {f['count_mae']:.2f}")
    print(f"  best count MAE : {b['count_mae']:.2f} (bias {b['count_bias']:+.2f}, "
          f"r {b['count_pearson_r']}) @ conf {b['conf']}  | F1 {b['f1']:.3f}")
    print(f"  totals @ best-MAE conf: pred {b['total_pred']} vs gt {b['total_gt']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"  full sweep -> {args.out}")


if __name__ == "__main__":
    main()
