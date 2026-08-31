"""Copy the winning model out of runs/comparison.json into ../Eval/models.

python export_best.py [--by f1|count_mae]
"""

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPARISON = ROOT / "runs/comparison.json"
DEST_DIR = ROOT.parent / "Eval/models"

WEIGHTS_FOR = {
    "C1": "runs/detect/merged_11s_1024_ep100_base/weights/best.pt",
    "C2": "runs/detect/merged_v8sP2_1024_ep200_histoaug/weights/best.pt",
    "C3": "runs/detect/merged_v8mP2_1024_ep200_histoaug/weights/best.pt",
    "D": "runs/heatmap_unet/best.pt",
    "baseline": "runs/detect/yolov11s_1024_dataO1_ep100_Noaug_b16/weights/best.pt",
}


def weights_for(model_label: str) -> Path | None:
    key = model_label.split()[0]
    rel = WEIGHTS_FOR.get(key)
    return ROOT / rel if rel else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["f1", "count_mae"], default="f1")
    args = ap.parse_args()

    rows = json.loads(COMPARISON.read_text())
    best = (max(rows, key=lambda r: r["f1"]) if args.by == "f1"
            else min(rows, key=lambda r: r["count_mae"]))
    src = weights_for(best["model"])
    if src is None or not src.exists():
        raise SystemExit(f"no weights for {best['model']!r}")

    slug = re.sub(r"[^a-z0-9]+", "_", best["model"].lower()).strip("_")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEST_DIR / f"best_{slug}{src.suffix}"
    shutil.copy2(src, dest)

    # metrics only hold at this imgsz/tta, so ship them with the weights
    m = re.search(r"@(\d+)", best["model"])
    meta = {
        "source_weights": str(src.relative_to(ROOT)),
        "selected_by": args.by,
        "test_slides": ["Aswcm3", "Sahar27"],
        "inference": {
            "imgsz": int(m.group(1)) if m else None,
            "tta": "+TTA" in best["model"],
            "threshold_for_best_f1": best["f1_conf"],
            "threshold_for_best_count": best["mae_conf"],
            "note": "for the heatmap model the threshold is a peak height, not a conf",
        },
        "metrics": best,
    }
    (dest.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"exported {best['model']}\n  weights -> {dest}\n  metadata -> {dest.with_suffix('.json')}")
    print(f"  F1 {best['f1']:.3f} @ conf {best['f1_conf']}, "
          f"count MAE {best['count_mae']:.2f} @ conf {best['mae_conf']}")


if __name__ == "__main__":
    main()
