"""UNet heatmap counter. Only needs box centers, so box-size noise doesn't matter.

python heatmap_counter.py train | eval [--tta]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader, Dataset

import eval_counting as ec

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data/Yolo_Merged/dataset"
CKPT = ROOT / "runs/heatmap_unet/best.pt"

IMG_SIZE = 1024
CROP = 512
SIGMA = 6.0
POS_WEIGHT = 10.0
BATCH = int(os.environ.get("HM_BATCH", 4))
EPOCHS = int(os.environ.get("HM_EPOCHS", 60))
LR = 3e-4
PEAK_MIN = 0.05  # floor only, real threshold comes from the sweep

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def render_heatmap(centers_px: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    hm = np.zeros((h, w), dtype=np.float32)
    if len(centers_px) == 0:
        return hm
    r = int(3 * SIGMA)
    ax = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * SIGMA ** 2))
    for cx, cy in centers_px:
        x, y = int(round(cx)), int(round(cy))
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx0, gy0 = x0 - (x - r), y0 - (y - r)
        patch = g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], patch)
    return hm


class TileDataset(Dataset):
    def __init__(self, split: str, train: bool):
        self.paths = sorted((DATA / "images" / split).glob("*.png"))
        self.labels = DATA / "labels" / split
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        pil = Image.open(p).convert("RGB")
        img = np.asarray(pil, dtype=np.float32) / 255.0
        centers = ec.load_gt_centers(self.labels / p.name.replace(".png", ".txt"),
                                     pil.size)
        h, w = img.shape[:2]
        hm = render_heatmap(centers, (h, w))
        if self.train:
            rng = np.random.default_rng()
            y0 = int(rng.integers(0, max(h - CROP, 0) + 1))
            x0 = int(rng.integers(0, max(w - CROP, 0) + 1))
            img, hm = img[y0:y0 + CROP, x0:x0 + CROP], hm[y0:y0 + CROP, x0:x0 + CROP]
            k = int(rng.integers(4))  # dihedral
            img, hm = np.rot90(img, k, (0, 1)), np.rot90(hm, k, (0, 1))
            if rng.random() < 0.5:
                img, hm = img[:, ::-1], hm[:, ::-1]
        img = (img - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(hm))[None])


def build_model():
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)


def weighted_mse(pred, target):
    # peaks are a tiny fraction of pixels, so upweight them
    return (F.mse_loss(pred, target, reduction="none")
            * (1.0 + POS_WEIGHT * target)).mean()


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    ds = TileDataset("train", train=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(dl))
    scaler = torch.amp.GradScaler(device)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for img, hm in dl:
            img, hm = img.to(device, non_blocking=True), hm.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device):
                loss = weighted_mse(model(img).sigmoid(), hm)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        avg = tot / len(dl)
        print(f"epoch {ep:3d}/{EPOCHS}  train loss {avg:.5f}", flush=True)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), CKPT)
    print(f"best train loss {best:.5f} -> {CKPT}")


@torch.no_grad()
def predict_centers(model, device, images_dir: Path, tta: bool = False) -> dict:
    model.eval()
    preds = {}
    for p in sorted(images_dir.glob("*.png")):
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        img = (img - MEAN) / STD
        x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
        with torch.amp.autocast(device):
            hm = model(x).sigmoid()[0, 0].float()
            if tta:
                for k in (1, 2, 3):
                    hm += torch.rot90(
                        model(torch.rot90(x, k, (2, 3))).sigmoid(), -k, (2, 3)
                    )[0, 0].float()
                hm /= 4
        hm = hm.cpu().numpy()
        peaks = (hm == maximum_filter(hm, size=9)) & (hm >= PEAK_MIN)
        ys, xs = np.nonzero(peaks)
        preds[p.name] = (np.stack([xs, ys], axis=1).astype(float),
                         hm[ys, xs].astype(float))
    return preds


def evaluate(tta: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    images_dir, labels_dir = DATA / "images/test", DATA / "labels/test"
    gts = ec.load_gt_for_images(images_dir, labels_dir)
    preds = predict_centers(model, device, images_dir, tta=tta)
    report = ec.sweep(preds, gts, confs=np.round(np.arange(0.05, 0.96, 0.05), 2))
    report["model"] = f"heatmap_unet(resnet34, sigma={SIGMA})"
    report["tta"] = tta
    b, f = report["best_count_mae"], report["best_f1"]
    print(f"heatmap UNet (tta={tta}, {len(gts)} tiles)")
    print(f"  best F1        : {f['f1']:.3f} (P {f['precision']:.3f} / R {f['recall']:.3f}) "
          f"@ thr {f['conf']}  | count MAE {f['count_mae']:.2f}")
    print(f"  best count MAE : {b['count_mae']:.2f} (bias {b['count_bias']:+.2f}, "
          f"r {b['count_pearson_r']}) @ thr {b['conf']}  | F1 {b['f1']:.3f}")
    out = ROOT / f"runs/eval_heatmap_unet{'_tta' if tta else ''}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  full sweep -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    (train if args.mode == "train" else lambda: evaluate(args.tta))()
