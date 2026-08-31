"""Merge Observer1+Observer2 into Data/Yolo_Merged, split by slide, write CV folds."""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
O1 = ROOT / "Data/Yolo_Observer1/dataset"
O2 = ROOT / "Data/Yolo_Observer2/dataset"
OUT = ROOT / "Data/Yolo_Merged/dataset"

MIN_WH = 2.0 / 1024
WBF_IOU = 0.3
MATCH_DIST_PX = 16.0
IMG_SIZE = 1024

TEST_SLIDES = {"Aswcm3", "Sahar27"}
N_FOLDS = 5


def slide_of(name: str) -> str:
    return name.split("tile")[0]


def load_labels(path: Path) -> np.ndarray:
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            x, y, w, h = map(float, parts[1:5])
            if w < MIN_WH or h < MIN_WH:
                continue
            x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
            w, h = min(w, 1.0), min(h, 1.0)
            boxes.append([x, y, w, h])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return np.clip(out, 0.0, 1.0)


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = (b[:, 0] + b[:, 2]) / 2
    out[:, 1] = (b[:, 1] + b[:, 3]) / 2
    out[:, 2] = b[:, 2] - b[:, 0]
    out[:, 3] = b[:, 3] - b[:, 1]
    return out


def fuse(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    if len(b1) == 0 and len(b2) == 0:
        return np.zeros((0, 4))
    boxes = [xywh_to_xyxy(b).tolist() for b in (b1, b2)]
    scores = [[1.0] * len(b1), [1.0] * len(b2)]
    labels = [[0] * len(b1), [0] * len(b2)]
    fused, _, _ = weighted_boxes_fusion(
        boxes, scores, labels, iou_thr=WBF_IOU, skip_box_thr=0.0
    )
    return xyxy_to_xywh(np.array(fused).reshape(-1, 4))


def agreement(b1: np.ndarray, b2: np.ndarray, thr_px: float) -> tuple[int, int, int]:
    n1, n2 = len(b1), len(b2)
    if n1 == 0 or n2 == 0:
        return 0, n1, n2
    c1, c2 = b1[:, :2] * IMG_SIZE, b2[:, :2] * IMG_SIZE
    dist = np.linalg.norm(c1[:, None] - c2[None, :], axis=2)
    ri, ci = linear_sum_assignment(dist)
    return int((dist[ri, ci] <= thr_px).sum()), n1, n2


def main():
    img_path = {}
    o1_tiles, o2_tiles = set(), set()
    for src, tiles in ((O1, o1_tiles), (O2, o2_tiles)):
        for split in ("train", "val"):
            for p in sorted((src / "images" / split).glob("*.png")):
                tiles.add(p.name)
                img_path.setdefault(p.name, p)

    def label_path(src: Path, name: str) -> Path:
        stem = name.replace(".png", ".txt")
        for split in ("train", "val"):
            p = src / "labels" / split / stem
            if p.exists():
                return p
        return src / "labels" / "train" / stem

    shared = o1_tiles & o2_tiles
    all_tiles = o1_tiles | o2_tiles

    tp = fp = fn = 0
    counts1, counts2 = [], []
    for name in sorted(shared):
        b1 = load_labels(label_path(O1, name))
        b2 = load_labels(label_path(O2, name))
        m, n1, n2 = agreement(b1, b2, MATCH_DIST_PX)
        tp += m
        fp += n1 - m
        fn += n2 - m
        counts1.append(n1)
        counts2.append(n2)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    counts1, counts2 = np.array(counts1), np.array(counts2)
    count_r = float(np.corrcoef(counts1, counts2)[0, 1]) if len(counts1) > 1 else 0.0
    count_mae = float(np.abs(counts1 - counts2).mean())
    print(f"O1 vs O2 on {len(shared)} shared tiles (<= {MATCH_DIST_PX:.0f}px):")
    print(f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    print(f"  count MAE={count_mae:.2f}  r={count_r:.3f}")

    merged = {}
    n_from = defaultdict(int)
    for name in sorted(all_tiles):
        if name in shared:
            b = fuse(load_labels(label_path(O1, name)), load_labels(label_path(O2, name)))
            n_from["fused"] += len(b)
        elif name in o2_tiles:
            b = load_labels(label_path(O2, name))
            n_from["o2_only"] += len(b)
        else:
            b = load_labels(label_path(O1, name))
            n_from["o1_only"] += len(b)
        merged[name] = b

    if OUT.exists():
        shutil.rmtree(OUT)
    train_slides = sorted({slide_of(n) for n in all_tiles} - TEST_SLIDES)
    split_of = {n: ("test" if slide_of(n) in TEST_SLIDES else "train") for n in all_tiles}

    for split in ("train", "test"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
    for name, boxes in merged.items():
        split = split_of[name]
        dst_img = OUT / "images" / split / name
        try:
            dst_img.symlink_to(img_path[name].resolve())  # symlink, disk is tight
        except OSError:
            shutil.copy2(img_path[name], dst_img)
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in boxes]
        (OUT / "labels" / split / name.replace(".png", ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    n_train = sum(1 for n in all_tiles if split_of[n] == "train")
    n_test = len(all_tiles) - n_train
    print(f"\n{OUT}")
    print(f"  tiles: {len(all_tiles)} (train {n_train} / {len(train_slides)} slides, "
          f"test {n_test} / {sorted(TEST_SLIDES)})")
    print(f"  boxes: {sum(len(b) for b in merged.values())} ({dict(n_from)})")

    # abs path: ultralytics resolves relative ones against its own datasets dir
    (ROOT / "neutrophils_merged.yaml").write_text(
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val: images/test\n\n"
        "names:\n  0: neutrophil\n"
    )

    rng = np.random.default_rng(42)
    order = rng.permutation(train_slides)
    folds = [sorted(order[i::N_FOLDS].tolist()) for i in range(N_FOLDS)]
    cv_dir = OUT / "cv"
    cv_dir.mkdir()
    train_tiles = [n for n in sorted(all_tiles) if split_of[n] == "train"]
    for i, val_slides in enumerate(folds, 1):
        fdir = cv_dir / f"fold_{i}"
        for split in ("train", "val"):
            (fdir / "images" / split).mkdir(parents=True)
            (fdir / "labels" / split).mkdir(parents=True)
        for name in train_tiles:
            split = "val" if slide_of(name) in val_slides else "train"
            (fdir / "images" / split / name).symlink_to(
                (OUT / "images/train" / name).resolve()
            )
            (fdir / "labels" / split / name.replace(".png", ".txt")).symlink_to(
                (OUT / "labels/train" / name.replace(".png", ".txt")).resolve()
            )
        (fdir / "fold.yaml").write_text(
            f"path: {fdir}\ntrain: images/train\nval: images/val\n\n"
            "names:\n  0: neutrophil\n"
        )
        print(f"  fold_{i}: val {val_slides}")

    (OUT / "summary.json").write_text(json.dumps({
        "interobserver": {"precision": prec, "recall": rec, "f1": f1,
                          "count_mae": count_mae, "count_pearson_r": count_r,
                          "match_dist_px": MATCH_DIST_PX},
        "test_slides": sorted(TEST_SLIDES),
        "train_slides": train_slides,
        "folds": {f"fold_{i+1}": f for i, f in enumerate(folds)},
    }, indent=2))


if __name__ == "__main__":
    main()
