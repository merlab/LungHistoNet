"""Count neutrophils per tile for every slide in the group dirs, box them, write CSVs.

python predict_slides.py [--root DIR] [--out DIR] [--imgsz N] [--conf C] [--no-images]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent
MODEL = REPO.parent / "Eval/models/best_c1_merged_11s_1536.pt"
SIDECAR = MODEL.with_suffix(".json")
GROUPS = ["EV P4 H and E stain", "EV pf H and E stain d14", "p5 h and e"]


def predict_batch(model, paths, imgsz, conf, bs=4):
    out = []
    i = 0
    while i < len(paths):
        chunk = paths[i:i + bs]
        try:
            res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf,
                                iou=0.5, max_det=2000, verbose=False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = max(1, bs // 2)  # gpu is shared
                continue
            raise
        out.extend(zip(chunk, res))
        i += len(chunk)
    return out


def stats(counts):
    c = np.asarray(counts, dtype=float)
    if not len(c):
        return dict(n_tiles=0, total=0, mean=0.0, median=0.0, std=0.0,
                    min=0, max=0, pct_tiles_with_cells=0.0)
    return dict(
        n_tiles=len(c), total=int(c.sum()), mean=round(float(c.mean()), 3),
        median=round(float(np.median(c)), 1), std=round(float(c.std(ddof=0)), 3),
        min=int(c.min()), max=int(c.max()),
        pct_tiles_with_cells=round(float((c > 0).mean() * 100), 1),
    )


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    meta = json.loads(SIDECAR.read_text())["inference"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/storage/Amir/Tiles_1024x1024")
    ap.add_argument("--out", default="/media/storage/Amir/neutrophil_predictions")
    ap.add_argument("--imgsz", type=int, default=meta["imgsz"])
    ap.add_argument("--conf", type=float, default=meta["threshold_for_best_count"])
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    print(f"model {MODEL.name}  imgsz={args.imgsz}  conf={args.conf}", flush=True)

    slide_rows, group_rows = [], []
    for group in GROUPS:
        gdir = root / group
        if not gdir.is_dir():
            print(f"[warn] missing group {group}")
            continue
        g_counts = []
        for sdir in sorted(p for p in gdir.iterdir() if p.is_dir()):
            tiles = sorted((sdir / "kept").glob("*.png"))
            gout = out / group / sdir.name
            gout.mkdir(parents=True, exist_ok=True)
            if not tiles:
                print(f"  {group}/{sdir.name}: no kept tiles, skipped", flush=True)
                slide_rows.append(dict(group=group, slide=sdir.name, **stats([])))
                continue
            img_dir = gout / "boxes"
            if not args.no_images:
                img_dir.mkdir(exist_ok=True)

            rows = []
            for p, r in predict_batch(model, tiles, args.imgsz, args.conf):
                n = len(r.boxes)
                if not args.no_images:
                    Image.fromarray(
                        r.plot(labels=False, conf=False, line_width=2)[:, :, ::-1]
                    ).save(img_dir / p.name)
                cf = r.boxes.conf.cpu().numpy() if n else np.zeros(0)
                xy = r.boxes.xywh.cpu().numpy() if n else np.zeros((0, 4))
                rows.append(dict(
                    tile=p.name, count=n,
                    mean_conf=round(float(cf.mean()), 3) if n else 0.0,
                    median_box_w=round(float(np.median(xy[:, 2])), 1) if n else 0.0,
                    median_box_h=round(float(np.median(xy[:, 3])), 1) if n else 0.0,
                ))
            write_csv(gout / f"{sdir.name}_tiles.csv", rows,
                      ["tile", "count", "mean_conf", "median_box_w", "median_box_h"])

            counts = [r["count"] for r in rows]
            g_counts += counts
            st = stats(counts)
            slide_rows.append(dict(group=group, slide=sdir.name, **st))
            print(f"  {group}/{sdir.name}: {st['n_tiles']} tiles, "
                  f"{st['total']} cells, {st['mean']:.2f}/tile", flush=True)

        group_rows.append(dict(group=group, n_slides=sum(
            1 for r in slide_rows if r["group"] == group and r["n_tiles"]), **stats(g_counts)))

    write_csv(out / "per_slide_summary.csv", slide_rows,
              ["group", "slide"] + list(stats([]).keys()))
    write_csv(out / "per_group_summary.csv", group_rows,
              ["group", "n_slides"] + list(stats([]).keys()))

    print("\n=== per group ===")
    for r in group_rows:
        print(f"{r['group']:26s} {r['n_slides']:2d} slides  {r['n_tiles']:5d} tiles  "
              f"{r['total']:6d} cells  {r['mean']:6.2f}/tile (sd {r['std']:.2f})  "
              f"{r['pct_tiles_with_cells']:5.1f}% tiles +ve")
    allc = sum(r["total"] for r in group_rows)
    allt = sum(r["n_tiles"] for r in group_rows)
    print(f"\nOVERALL: {allt} tiles, {allc} cells, {allc/max(allt,1):.2f}/tile")
    print(f"csvs -> {out}")


if __name__ == "__main__":
    main()
