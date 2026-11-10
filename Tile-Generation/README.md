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

## Fixed physical scale (`--target-mpp`)

By default tiles are cropped at pyramid level 0 in raw pixels, so a `--patch-w 256`
tile covers a different amount of *tissue* on every slide whose native resolution
differs. This is not hypothetical: across 317 TCGA-BRCA diagnostic slides the native
resolution is 0.25 um/px on 274, 0.23 on 20, 0.5 on 14 and 0.16 on 5. A fixed pixel
crop makes 14 of those slides cover 2x the physical field of view of the rest, which
is a per-slide scale artifact any downstream model can key on.

`--target-mpp` fixes the *physical* tile size instead. The script reads the nearest
pyramid level that is still finer than the target, crops the corresponding number of
source pixels, and resamples (Lanczos) down to `--patch-w/--patch-h`:

```bash
python generate_tiles.py --mode grid \
  --slide-dir "/path/to/slides/TCGA-BRCA" --recursive \
  --output-base "/path/to/tiles_256px_0.5mpp/BRCA" \
  --patch-w 256 --patch-h 256 --target-mpp 0.5 \
  --max-tiles-per-slide 2000 --seed 0 \
  --no-save-rejected --prefer-ext svs
```

That yields 256x256 tiles at 0.5 um/px = **128 um of tissue per tile (20x)** on every
slide. Match the target to whatever consumes the tiles: pathology foundation models
such as UNI2-h and Virchow2 take 224x224 input and are pretrained near 0.5 um/px, so
tiles at 0.5 um/px reach them at their native scale instead of being downsampled.

Partial edge crops are dropped rather than stretched, since stretching one to the full
output size would give it a different um/px than every other tile.

Each slide folder gets a `tiling_manifest.json` recording native MPP, the level read,
crop and output sizes, microns per tile, the cap and the seed - without it the physical
scale of a tile set is unrecoverable after the fact. A slide folder that already has a
manifest is skipped, so re-running resumes. Slides that fail to open are reported at the
end instead of aborting the batch.

| Flag | Description |
|------|-------------|
| `--target-mpp 0.5` | Resample every tile to this microns-per-pixel |
| `--assume-mpp 0.25` | MPP for slides with no resolution metadata (warns when used) |
| `--max-tiles-per-slide 2000` | Cap kept tiles/slide; candidates are shuffled first, so the cap is a spatially uniform sample over tissue (may overshoot by <64) |
| `--seed 0` | Seed for that sampling |
| `--recursive` | Walk `<project>/<patient>/<slide>.svs` trees |

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
