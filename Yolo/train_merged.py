"""Training runs on the merged data. Skips whatever is already done."""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = str(ROOT / "neutrophils_merged.yaml")

# no canonical orientation in histology; erasing can wipe out whole cells
HISTO_AUGS = dict(flipud=0.5, fliplr=0.5, degrees=90, erasing=0.0,
                  scale=0.3, close_mosaic=15)

RUNS = [
    dict(name="merged_11s_1024_ep100_base",
         model="yolo11s.pt", cfg=None,
         train=dict(epochs=100, imgsz=1024, batch=12)),
    dict(name="merged_v8sP2_1024_ep200_histoaug",
         model="yolov8s.pt", cfg="yolov8s-p2.yaml",  # p2 = stride-4 head
         train=dict(epochs=200, imgsz=1024, batch=8, patience=40,
                    cos_lr=True, **HISTO_AUGS)),
    dict(name="merged_v8mP2_1024_ep200_histoaug",
         model="yolov8m.pt", cfg="yolov8m-p2.yaml",
         train=dict(epochs=200, imgsz=1024, batch=6, patience=40,
                    cos_lr=True, **HISTO_AUGS)),
]


def main():
    for spec in RUNS:
        best = ROOT / "runs/detect" / spec["name"] / "weights/best.pt"
        if best.exists():
            print(f"[skip] {spec['name']}")
            continue
        print(f"\n===== training {spec['name']} =====", flush=True)
        model = YOLO(spec["cfg"]).load(spec["model"]) if spec["cfg"] else YOLO(spec["model"])
        model.to("cuda")
        try:
            model.train(data=DATA, name=spec["name"], exist_ok=True, plots=True,
                        **spec["train"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch = max(2, spec["train"]["batch"] // 2)
            print(f"[oom] retry {spec['name']} @ batch={batch}", flush=True)
            spec["train"]["batch"] = batch
            model = YOLO(spec["cfg"]).load(spec["model"]) if spec["cfg"] else YOLO(spec["model"])
            model.train(data=DATA, name=spec["name"], exist_ok=True, plots=True,
                        **spec["train"])
        del model
        torch.cuda.empty_cache()
    print("\nall training runs finished")


if __name__ == "__main__":
    main()
