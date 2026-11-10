"""Score every trained variant on the held-out slides, one table.

python compare_models.py [--quick]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import eval_counting as ec

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data/Yolo_Merged/dataset"
OUT_JSON = ROOT / "runs/comparison.json"


def yolo_variants(quick: bool):
    runs = [
        ("baseline O1-only (old)", "runs/detect/yolov11s_1024_dataO1_ep100_Noaug_b16/weights/best.pt"),
        ("C1 merged 11s", "runs/detect/merged_11s_1024_ep100_base/weights/best.pt"),
        ("C2 merged v8s-P2 +augs", "runs/detect/merged_v8sP2_1024_ep200_histoaug/weights/best.pt"),
        ("C3 merged v8m-P2 +augs", "runs/detect/merged_v8mP2_1024_ep200_histoaug/weights/best.pt"),
    ]
    settings = [(1024, False)] if quick else [(1024, False), (1536, False), (1024, True)]
    for label, w in runs:
        if not (ROOT / w).exists():
            continue
        for imgsz, tta in settings:
            tag = f"{label} @{imgsz}{' +TTA' if tta else ''}"
            yield tag, str(ROOT / w), imgsz, tta


def row_from_report(tag: str, rep: dict) -> dict:
    b, f = rep["best_count_mae"], rep["best_f1"]
    return {
        "model": tag,
        "f1": f["f1"], "precision": f["precision"], "recall": f["recall"],
        "f1_conf": f["conf"],
        "count_mae": b["count_mae"], "count_bias": b["count_bias"],
        "count_r": b["count_pearson_r"], "mae_conf": b["conf"],
        "total_pred": b["total_pred"], "total_gt": b["total_gt"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    images_dir, labels_dir = DATA / "images/test", DATA / "labels/test"
    gts = ec.load_gt_for_images(images_dir, labels_dir)
    rows = []

    for tag, weights, imgsz, tta in yolo_variants(args.quick):
        print(f"evaluating {tag} ...", flush=True)
        preds = ec.yolo_predict_centers(weights, images_dir, tta=tta, imgsz=imgsz)
        rows.append(row_from_report(tag, ec.sweep(preds, gts)))

    hm_ckpt = ROOT / "runs/heatmap_unet/best.pt"
    if hm_ckpt.exists():
        import torch
        import heatmap_counter as hc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = hc.build_model().to(device)
        model.load_state_dict(torch.load(hm_ckpt, map_location=device, weights_only=True))
        for tta in ([False] if args.quick else [False, True]):
            tag = f"D heatmap UNet{' +TTA' if tta else ''}"
            print(f"evaluating {tag} ...", flush=True)
            preds = hc.predict_centers(model, device, images_dir, tta=tta)
            rows.append(row_from_report(
                tag, ec.sweep(preds, gts, confs=np.round(np.arange(0.05, 0.96, 0.05), 2))))

    rows.sort(key=lambda r: -r["f1"])
    w = max(len(r["model"]) for r in rows) + 2
    print(f"\n{'model':<{w}}{'F1':>7}{'P':>7}{'R':>7}{'@conf':>7}"
          f"{'MAE':>8}{'bias':>8}{'r':>7}{'@conf':>7}")
    print("-" * (w + 58))
    for r in rows:
        print(f"{r['model']:<{w}}{r['f1']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['f1_conf']:>7.2f}{r['count_mae']:>8.2f}{r['count_bias']:>+8.2f}"
              f"{(r['count_r'] or 0):>7.3f}{r['mae_conf']:>7.2f}")
    print(f"\ntest set: {len(gts)} tiles, {sum(len(v) for v in gts.values())} neutrophils "
          f"(slides Aswcm3, Sahar27)")
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
