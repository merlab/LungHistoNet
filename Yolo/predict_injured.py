"""Box every tile in a folder with the exported model, dump counts.

python predict_injured.py [--source test/injured] [--imgsz N] [--conf C]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "Eval/models/best_c1_merged_11s_1536.pt"
SIDECAR = MODEL.with_suffix(".json")


def main():
    meta = json.loads(SIDECAR.read_text())["inference"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test/injured")
    ap.add_argument("--imgsz", type=int, default=meta["imgsz"])
    ap.add_argument("--conf", type=float, default=meta["threshold_for_best_count"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = ROOT / args.source
    out = Path(args.out) if args.out else ROOT / f"runs/predict_{src.name}"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL))
    images = sorted(src.glob("*.png"))
    rows = []
    for i in range(0, len(images), 4):
        chunk = images[i:i + 4]
        results = model.predict([str(p) for p in chunk], imgsz=args.imgsz,
                                conf=args.conf, iou=0.5, max_det=2000, verbose=False)
        for p, r in zip(chunk, results):
            Image.fromarray(r.plot(labels=False, conf=False, line_width=2)[:, :, ::-1]) \
                 .save(out / p.name)  # plot gives BGR
            wh = r.boxes.xywh.cpu().numpy()[:, 2:] if len(r.boxes) else np.zeros((0, 2))
            conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            rows.append({
                "tile": p.name,
                "count": len(r.boxes),
                "mean_conf": round(float(conf.mean()), 3) if len(conf) else 0.0,
                "median_box_w": round(float(np.median(wh[:, 0])), 1) if len(wh) else 0.0,
                "median_box_h": round(float(np.median(wh[:, 1])), 1) if len(wh) else 0.0,
            })

    csv_path = out / "counts.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts = np.array([r["count"] for r in rows])
    print(f"model {MODEL.name}  imgsz={args.imgsz}  conf={args.conf}")
    print(f"{len(rows)} tiles -> {out}")
    print(f"total neutrophils: {counts.sum()}")
    print(f"per tile: mean {counts.mean():.1f}, median {np.median(counts):.0f}, "
          f"min {counts.min()}, max {counts.max()}")
    print(f"counts CSV -> {csv_path}")


if __name__ == "__main__":
    main()
