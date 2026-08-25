# Tile Generation

Generate 1024×1024 (or custom size) tiles from whole-slide images (`.svs`, `.mrxs`, etc.) with tissue-aware filtering.

## Setup

```bash
conda activate tf
export PYTHONNOUSERSITE=1   # avoids user-site numpy conflicts
```

Requires: `pyvips`, `opencv-python`, `numpy`, `tqdm` (in the `tf` env).

## Basic usage (grid mode)

Process all slides in a folder:

```bash
python generate_tiles.py --mode grid \
  --slide-dir "~/path/to/slides/" \
  --patch-w 1024 --patch-h 1024 \
  --prefer-ext svs
```

### Output layout (default)

Tiles are written under `<parent-of-slide-dir>/Tiles_{h}x{w}/`, or use `--output-base` to set the root:

```
Tiles_1024x1024/
  EV P4 H and E stain/
    P4 M114/
      kept/                      # passed filters (PNG)
      discarded/blurry/            # rejected, by reason (JPEG)
      discarded/large_background/
      ...
      reject_summary.txt
```

### Example: New Slides Eva

```bash
BASE="$HOME/Documents/Data/ALI surgical/New Slides Eva"

for sub in "EV P4 H and E stain" "EV pf H and E stain d14" "p5 h and e"; do
  python generate_tiles.py --mode grid \
    --slide-dir "$BASE/$sub" \
    --output-base "$BASE/Tiles_1024x1024" \
    --patch-w 1024 --patch-h 1024 \
    --prefer-ext svs
done
```

## Useful flags

| Flag | Description |
|------|-------------|
| `--output-base PATH` | Root output directory (default: `<parent>/Tiles_{h}x{w}/`) |
| `--prefer-ext svs` | Use `.svs` when both `.svs` and `.mrxs` exist |
| `--overlap 0.15` | Grid overlap fraction (0–0.9) |
| `--no-save-rejected` | Only write kept tiles (flat layout, no `discarded/`) |
| `--save-rejected-mask` | Also save mask-skipped tiles (large disk use) |

Quality tuning (optional): `--min-blur-var`, `--min-tissue-frac`, `--min-edge-frac`, etc. Run `--help` for full list.

## Other modes

**Random** — sample random tiles from one slide:

```bash
python generate_tiles.py --mode random \
  --image "~/path/to/slide.svs" \
  --patch-w 1024 --patch-h 1024 \
  --patch-type fixed --num-tiles 200
```

**Stitch** — rebuild an image from saved tiles:

```bash
python generate_tiles.py --mode stitch \
  --tile-dir "~/path/to/slide/kept" \
  --slide "~/path/to/slide.svs" \
  --patch-w 1024 --patch-h 1024
```

## Help

```bash
python generate_tiles.py --help
```
