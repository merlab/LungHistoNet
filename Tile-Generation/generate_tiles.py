"""
Tile generation for whole-slide images (.mrxs, .svs, and other OpenSlide formats).

Grid output layout (default):
  {output_base}/{slide_folder_name}/{slide_stem}/
    kept/                    # tiles that passed quality filters
    discarded/{reason}/      # rejected tiles grouped by reason
    reject_summary.txt       # counts per category
"""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import pyvips
from tqdm import tqdm

# Formats commonly opened via libvips/OpenSlide
SLIDE_EXTENSIONS = (".mrxs", ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".bif")


# ---------------------------------------------------------------------------
# Quality / tissue detection
# ---------------------------------------------------------------------------

@dataclass
class QualityConfig:
    """Filters to prefer tissue over blank / artifact tiles."""

    lower_intensity: int = 240
    upper_intensity: int = 255
    use_intensity_filter: bool = True

    # HSV tissue pixel rules (OpenCV H:0-179, S/V:0-255)
    sat_min: int = 24          # S > sat_min  OR
    value_max: int = 238       # V < value_max  => tissue-ish
    min_tissue_frac: float = 0.45
    # When raw tissue frac is low, still keep if white is mostly internal holes
    # (alveoli) rather than one large border-connected blank region.
    min_filled_tissue_frac: float = 0.75
    max_border_bg_frac: float = 0.20

    # Reject washed-out / haze tiles with little real stain
    min_sat_mean: float = 15.0

    # Reject out-of-focus / structureless tiles
    min_blur_var: float = 32.0
    use_blur_filter: bool = True
    min_edge_frac: float = 0.008  # Canny edge density; blurry haze ~0
    min_gray_std: float = 30.0    # low contrast blank/haze

    # Reject extreme red/blue pen marks (OpenCV hue)
    reject_pen: bool = True
    pen_frac_max: float = 0.08

    # Slide-level mask
    use_tissue_mask: bool = True
    mask_max_dim: int = 2048
    mask_min_tile_frac: float = 0.35
    morph_kernel: int = 5


def ensure_rgb(tile_array: np.ndarray) -> np.ndarray:
    """Return HxWx3 uint8 RGB (drop alpha if present)."""
    if tile_array.ndim == 2:
        return cv.cvtColor(tile_array, cv.COLOR_GRAY2RGB)
    if tile_array.shape[2] == 1:
        return cv.cvtColor(tile_array, cv.COLOR_GRAY2RGB)
    if tile_array.shape[2] >= 3:
        return tile_array[:, :, :3].copy()
    raise ValueError(f"Unexpected tile shape: {tile_array.shape}")


def tissue_pixel_mask(rgb: np.ndarray, cfg: QualityConfig) -> np.ndarray:
    """Boolean mask of likely tissue pixels (not near-white background)."""
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return (s > cfg.sat_min) | (v < cfg.value_max)


def tissue_fraction(rgb: np.ndarray, cfg: QualityConfig) -> float:
    mask = tissue_pixel_mask(rgb, cfg)
    return float(np.mean(mask)) if mask.size else 0.0


def background_geometry(rgb: np.ndarray, cfg: QualityConfig):
    """
    Separate contiguous slide-background white from distributed internal white
    (e.g. alveolar air spaces).

    Flood-fills background from the tile border: border-connected white is
    true background; remaining white components are internal holes.

    Returns:
        border_bg_frac, internal_bg_frac, filled_tissue_frac
        where filled_tissue = tissue | internal_holes.
    """
    tissue = tissue_pixel_mask(rgb, cfg)
    bg = (~tissue).astype(np.uint8) * 255
    h, w = bg.shape
    filled = bg.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        x, y = seed
        if filled[y, x] == 255:
            cv.floodFill(filled, flood_mask, seed, 128)

    border_bg = filled == 128
    internal_bg = filled == 255
    filled_tissue = tissue | internal_bg
    return (
        float(np.mean(border_bg)),
        float(np.mean(internal_bg)),
        float(np.mean(filled_tissue)),
    )


def blur_variance(rgb: np.ndarray) -> float:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def edge_fraction(rgb: np.ndarray) -> float:
    """Fraction of Canny edge pixels — near 0 for blurry haze / blank."""
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def mean_saturation(rgb: np.ndarray) -> float:
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 1]))


def gray_std(rgb: np.ndarray) -> float:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    return float(np.std(gray))


def pen_mark_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels in extreme red or blue hue (pen / marker)."""
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Only count saturated, darkish marks (not pale stain)
    candidate = (s > 80) & (v < 220)
    red = ((h <= 10) | (h >= 170)) & candidate
    blue = (h >= 100) & (h <= 130) & candidate
    return float(np.mean(red | blue))


def is_blank_by_intensity(rgb: np.ndarray, cfg: QualityConfig) -> bool:
    mean_value = float(np.mean(rgb))
    return cfg.lower_intensity < mean_value <= cfg.upper_intensity


def tile_quality_decision(rgb: np.ndarray, cfg: QualityConfig):
    """
    Evaluate tile quality.

    Returns:
        (ok, reason, metrics) where reason is 'kept' or a reject label.
    """
    rgb = ensure_rgb(rgb)
    border_bg, internal_bg, filled_tissue = background_geometry(rgb, cfg)
    metrics = {
        "tissue_frac": tissue_fraction(rgb, cfg),
        "border_bg_frac": border_bg,
        "internal_bg_frac": internal_bg,
        "filled_tissue_frac": filled_tissue,
        "sat_mean": mean_saturation(rgb),
        "gray_std": gray_std(rgb),
        "blur_var": blur_variance(rgb),
        "edge_frac": edge_fraction(rgb),
        "pen_frac": pen_mark_fraction(rgb),
        "mean": float(np.mean(rgb)),
    }

    if cfg.use_intensity_filter and is_blank_by_intensity(rgb, cfg):
        return False, "intensity_blank", metrics

    # Low raw tissue fraction: keep if white is mostly internal (alveoli),
    # reject if a large contiguous border-connected blank dominates.
    if metrics["tissue_frac"] < cfg.min_tissue_frac:
        alveolar_like = (
            metrics["filled_tissue_frac"] >= cfg.min_filled_tissue_frac
            and metrics["border_bg_frac"] <= cfg.max_border_bg_frac
        )
        if not alveolar_like:
            if metrics["border_bg_frac"] > cfg.max_border_bg_frac:
                return False, "large_background", metrics
            return False, "low_tissue_frac", metrics

    if metrics["sat_mean"] < cfg.min_sat_mean:
        return False, "low_saturation", metrics
    if metrics["gray_std"] < cfg.min_gray_std:
        return False, "low_contrast", metrics
    if cfg.use_blur_filter and metrics["blur_var"] < cfg.min_blur_var:
        return False, "blurry", metrics
    if metrics["edge_frac"] < cfg.min_edge_frac:
        return False, "low_edges", metrics
    if cfg.reject_pen and metrics["pen_frac"] > cfg.pen_frac_max:
        return False, "pen_mark", metrics
    return True, "kept", metrics


def tile_passes_quality(rgb: np.ndarray, cfg: QualityConfig) -> bool:
    """Return True if the tile looks like usable tissue."""
    ok, _, _ = tile_quality_decision(rgb, cfg)
    return ok


def build_slide_tissue_mask(slide: pyvips.Image, cfg: QualityConfig, file_path: str | None = None):
    """
    Build a low-res binary tissue mask for the whole slide.

    Prefer pyvips thumbnail(file) so OpenSlide pyramid levels are used
    instead of resizing the full-resolution image.

    Returns:
        mask (H_m x W_m bool), scale_x, scale_y  (full_res = mask_coord * scale)
    """
    full_w, full_h = slide.width, slide.height

    if file_path:
        thumb = pyvips.Image.thumbnail(file_path, cfg.mask_max_dim)
    else:
        scale = min(1.0, cfg.mask_max_dim / max(full_w, full_h))
        thumb = slide.resize(scale) if scale < 1.0 else slide

    arr = ensure_rgb(pyvips_to_numpy(thumb))
    tissue = tissue_pixel_mask(arr, cfg).astype(np.uint8) * 255

    k = max(3, cfg.morph_kernel | 1)  # odd kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    tissue = cv.morphologyEx(tissue, cv.MORPH_OPEN, kernel)
    tissue = cv.morphologyEx(tissue, cv.MORPH_CLOSE, kernel)

    # Drop tiny speckles
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(tissue, connectivity=8)
    cleaned = np.zeros_like(tissue)
    min_area = max(32, int(0.0001 * tissue.shape[0] * tissue.shape[1]))
    for label in range(1, n_labels):
        if stats[label, cv.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    mask = cleaned > 0
    scale_x = full_w / mask.shape[1]
    scale_y = full_h / mask.shape[0]
    return mask, scale_x, scale_y


def mask_covers_tile(
    tissue_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    x: int,
    y: int,
    patch_w: int,
    patch_h: int,
    min_frac: float,
) -> bool:
    """True if enough of the tile bbox overlaps the slide-level tissue mask."""
    x0 = int(x / scale_x)
    y0 = int(y / scale_y)
    x1 = int((x + patch_w) / scale_x)
    y1 = int((y + patch_h) / scale_y)

    h, w = tissue_mask.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
    region = tissue_mask[y0:y1, x0:x1]
    if region.size == 0:
        return False
    return float(np.mean(region)) >= min_frac


def sample_points_from_mask(
    tissue_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    n: int,
    patch_w: int,
    patch_h: int,
    slide_w: int,
    slide_h: int,
):
    """Sample random top-left corners whose tile center lands on tissue."""
    ys, xs = np.where(tissue_mask)
    if len(xs) == 0:
        return []

    points = []
    max_attempts = max(n * 20, 1000)
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        idx = random.randint(0, len(xs) - 1)
        cx = int(xs[idx] * scale_x)
        cy = int(ys[idx] * scale_y)
        x = cx - patch_w // 2
        y = cy - patch_h // 2
        x = int(np.clip(x, 0, max(0, slide_w - patch_w)))
        y = int(np.clip(y, 0, max(0, slide_h - patch_h)))
        points.append((x, y))
    return points


def quality_config_from_args(args) -> QualityConfig:
    return QualityConfig(
        lower_intensity=args.lower_intensity,
        upper_intensity=args.upper_intensity,
        use_intensity_filter=not args.no_intensity_filter,
        sat_min=args.sat_min,
        value_max=args.value_max,
        min_tissue_frac=args.min_tissue_frac,
        min_filled_tissue_frac=args.min_filled_tissue_frac,
        max_border_bg_frac=args.max_border_bg_frac,
        min_sat_mean=args.min_sat_mean,
        min_blur_var=args.min_blur_var,
        use_blur_filter=not args.no_blur_filter,
        min_edge_frac=args.min_edge_frac,
        min_gray_std=args.min_gray_std,
        reject_pen=not args.no_pen_filter,
        pen_frac_max=args.pen_frac_max,
        use_tissue_mask=not args.no_tissue_mask,
        mask_max_dim=args.mask_max_dim,
        mask_min_tile_frac=args.mask_min_tile_frac,
        morph_kernel=args.morph_kernel,
    )


# ---------------------------------------------------------------------------
# Utilities and helpers
# ---------------------------------------------------------------------------

def highest_divisor(n):
    if n <= 1:
        return None

    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return n // i if n // i != n else i


def apply_gamma_correction(img, gamma):
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv.LUT(img, look_up_table)


def pyvips_to_numpy(vips_image):
    bands = vips_image.bands
    height = vips_image.height
    width = vips_image.width
    buf = vips_image.write_to_memory()
    if bands == 1:
        return np.ndarray(buffer=buf, dtype=np.uint8, shape=[height, width])
    return np.ndarray(buffer=buf, dtype=np.uint8, shape=[height, width, bands])


def _write_tile(tile, path: str, save_ext: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if save_ext.lower() == "png":
        tile.pngsave(path, compression=9)
    else:
        # JPEG is fine for visual QC of rejects and uses less disk
        tile.jpegsave(path, Q=85)


def process_tile(
    slide,
    width,
    height,
    x,
    y,
    patch_size_w,
    patch_size_h,
    fdir,
    cfg: QualityConfig,
    save_ext: str = "jpg",
    save_rejected: bool = False,
    reject_reason: str | None = None,
):
    """
    Crop and optionally save a tile.

    If reject_reason is set (e.g. 'mask_skip'), skip quality checks and save
    under discarded/<reason> when save_rejected is True.

    Returns dict: {status, reason, path} where status is 'kept'|'discarded'|'skipped'.
    """
    actual_patch_w = min(patch_size_w, width - x)
    actual_patch_h = min(patch_size_h, height - y)
    if actual_patch_w <= 0 or actual_patch_h <= 0:
        return {"status": "skipped", "reason": "oob", "path": None}

    tile = slide.crop(x, y, actual_patch_w, actual_patch_h)

    if reject_reason is not None:
        if not save_rejected:
            return {"status": "discarded", "reason": reject_reason, "path": None}
        out_ext = "jpg"
        path = os.path.join(fdir, "discarded", reject_reason, f"tile_{x}_{y}.{out_ext}")
        _write_tile(tile, path, out_ext)
        return {"status": "discarded", "reason": reject_reason, "path": path}

    tile_array = pyvips_to_numpy(tile)
    ok, reason, _metrics = tile_quality_decision(tile_array, cfg)

    if ok:
        if save_rejected:
            path = os.path.join(fdir, "kept", f"tile_{x}_{y}.{save_ext}")
        else:
            path = os.path.join(fdir, f"tile_{x}_{y}.{save_ext}")
        _write_tile(tile, path, save_ext)
        return {"status": "kept", "reason": "kept", "path": path}

    if save_rejected:
        out_ext = "jpg"
        path = os.path.join(fdir, "discarded", reason, f"tile_{x}_{y}.{out_ext}")
        _write_tile(tile, path, out_ext)
        return {"status": "discarded", "reason": reason, "path": path}

    return {"status": "discarded", "reason": reason, "path": None}


def process_tile_2(
    x,
    y,
    width,
    height,
    patch_size_w,
    patch_size_h,
    output_dir,
    slide,
    cfg: QualityConfig,
    save_rejected: bool = False,
    reject_reason: str | None = None,
):
    return process_tile(
        slide=slide,
        width=width,
        height=height,
        x=x,
        y=y,
        patch_size_w=patch_size_w,
        patch_size_h=patch_size_h,
        fdir=output_dir,
        cfg=cfg,
        save_ext="png" if not save_rejected else "png",
        save_rejected=save_rejected,
        reject_reason=reject_reason,
    )


# ---------------------------------------------------------------------------
# Random tile generation
# ---------------------------------------------------------------------------

def generate_random_tiles(
    image_path,
    patch_type,
    num_tiles,
    cfg: QualityConfig,
    output_dir=None,
    patch_w=None,
    patch_h=None,
    min_size=None,
    max_size=None,
    margin=800,
    max_attempts_factor: int = 20,
):
    """Generate random tiles from a single slide, preferring tissue regions."""
    image_path = os.path.expanduser(image_path)
    slide = pyvips.Image.new_from_file(image_path)
    width = slide.width
    height = slide.height

    if patch_type == "fixed":
        if patch_w is None or patch_h is None:
            raise ValueError("--patch-w and --patch-h are required for patch-type=fixed")
        fixed_w, fixed_h = patch_w, patch_h
        default_dir = f"Tile{fixed_h}{fixed_w}_{image_path[-6:-5]}"
        sample_w, sample_h = fixed_w, fixed_h
    else:
        if min_size is None or max_size is None:
            raise ValueError(
                "--min-size and --max-size are required for patch-type="
                f"{patch_type}"
            )
        if min_size > max_size:
            raise ValueError("--min-size must be <= --max-size")
        default_dir = f"Tile_{min_size}_{max_size}_{image_path[-6:-5]}"
        sample_w = sample_h = max_size

    if output_dir is None:
        # Same directory as the slide: <slide_dir>/<default_dir>
        fdir = os.path.join(os.path.dirname(image_path) or ".", default_dir)
    else:
        fdir = os.path.expanduser(output_dir)
    os.makedirs(fdir, exist_ok=True)

    tissue_mask = scale_x = scale_y = None
    if cfg.use_tissue_mask:
        tissue_mask, scale_x, scale_y = build_slide_tissue_mask(
            slide, cfg, file_path=image_path
        )
        if not np.any(tissue_mask):
            print("Warning: tissue mask is empty; falling back to uniform random sampling.")
            tissue_mask = None

    # Precompute candidate origins when mask is available (fixed size)
    candidates = None
    if tissue_mask is not None and patch_type == "fixed":
        candidates = sample_points_from_mask(
            tissue_mask,
            scale_x,
            scale_y,
            n=num_tiles * max_attempts_factor,
            patch_w=sample_w,
            patch_h=sample_h,
            slide_w=width,
            slide_h=height,
        )

    saved = 0
    attempts = 0
    max_attempts = num_tiles * max_attempts_factor
    candidate_idx = 0

    with ThreadPoolExecutor() as executor:
        while saved < num_tiles and attempts < max_attempts:
            batch = []
            while len(batch) < min(32, num_tiles - saved) and attempts < max_attempts:
                attempts += 1

                if patch_type == "square":
                    patch_size_w = patch_size_h = random.randint(min_size, max_size)
                elif patch_type == "rectangle":
                    patch_size_w = random.randint(min_size, max_size)
                    patch_size_h = random.randint(min_size, max_size)
                elif patch_type == "fixed":
                    patch_size_w, patch_size_h = fixed_w, fixed_h
                else:
                    raise ValueError(f"Unknown patch type: {patch_type}")

                if candidates:
                    x, y = candidates[candidate_idx % len(candidates)]
                    candidate_idx += 1
                elif tissue_mask is not None:
                    pts = sample_points_from_mask(
                        tissue_mask,
                        scale_x,
                        scale_y,
                        n=1,
                        patch_w=patch_size_w,
                        patch_h=patch_size_h,
                        slide_w=width,
                        slide_h=height,
                    )
                    if not pts:
                        continue
                    x, y = pts[0]
                else:
                    max_x = max(0, width - max(margin, patch_size_w))
                    max_y = max(0, height - max(margin, patch_size_h))
                    if width < patch_size_w or height < patch_size_h:
                        continue
                    x = random.randint(0, max_x) if max_x > 0 else 0
                    y = random.randint(0, max_y) if max_y > 0 else 0

                if tissue_mask is not None and not mask_covers_tile(
                    tissue_mask,
                    scale_x,
                    scale_y,
                    x,
                    y,
                    patch_size_w,
                    patch_size_h,
                    cfg.mask_min_tile_frac,
                ):
                    continue

                batch.append(
                    executor.submit(
                        process_tile,
                        slide,
                        width,
                        height,
                        x,
                        y,
                        patch_size_w,
                        patch_size_h,
                        fdir,
                        cfg,
                        "jpg",
                    )
                )

            for future in batch:
                result = future.result()
                if result and result.get("status") == "kept":
                    saved += 1
                    print(f"Saved: {result.get('path')}")
                    if saved >= num_tiles:
                        break

    print(f"Random tiling done: saved {saved}/{num_tiles} tiles ({attempts} attempts).")

# ---------------------------------------------------------------------------
# Fixed sliding-window tile generation
# ---------------------------------------------------------------------------

def generate_tiles(
    width,
    height,
    patch_size_w,
    patch_size_h,
    output_dir,
    slide,
    cfg: QualityConfig,
    overlap: float = 0.0,
    tissue_mask=None,
    scale_x=1.0,
    scale_y=1.0,
    save_rejected: bool = False,
    save_rejected_mask: bool = False,
):
    """Slide a tile window across the slide; keep tissue-quality tiles only."""
    overlap = float(np.clip(overlap, 0.0, 0.9))
    step_w = max(1, int(patch_size_w * (1.0 - overlap)))
    step_h = max(1, int(patch_size_h * (1.0 - overlap)))

    if save_rejected:
        os.makedirs(os.path.join(output_dir, "kept"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "discarded"), exist_ok=True)

    counts = {}
    mask_skipped = 0

    with ThreadPoolExecutor() as executor:
        futures = []
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                mask_ok = True
                if tissue_mask is not None and not mask_covers_tile(
                    tissue_mask,
                    scale_x,
                    scale_y,
                    x,
                    y,
                    patch_size_w,
                    patch_size_h,
                    cfg.mask_min_tile_frac,
                ):
                    mask_ok = False
                    mask_skipped += 1
                    if not (save_rejected and save_rejected_mask):
                        continue

                futures.append(
                    executor.submit(
                        process_tile_2,
                        x,
                        y,
                        width,
                        height,
                        patch_size_w,
                        patch_size_h,
                        output_dir,
                        slide,
                        cfg,
                        save_rejected,
                        None if mask_ok else "mask_skip",
                    )
                )

        for future in futures:
            result = future.result()
            reason = result.get("reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1

    kept = counts.get("kept", 0)
    if save_rejected:
        summary_path = os.path.join(output_dir, "reject_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"kept: {kept}\n")
            f.write(f"mask_skipped_not_saved: {mask_skipped if not save_rejected_mask else 0}\n")
            f.write(f"mask_skipped_total: {mask_skipped}\n")
            for reason, n in sorted(counts.items()):
                f.write(f"{reason}: {n}\n")
        print(f"  reject summary -> {summary_path}")
        for reason, n in sorted(counts.items()):
            print(f"    {reason}: {n}")

    return kept


def list_slide_files(slide_dir, prefer_ext: str | None = "svs"):
    """
    List slide files in a directory.

    If prefer_ext is set (e.g. 'svs') and both .svs and .mrxs exist for the
    same stem, keep only the preferred extension to avoid double-tiling.
    """
    files = [
        f for f in os.listdir(slide_dir)
        if f.lower().endswith(SLIDE_EXTENSIONS)
    ]
    if not prefer_ext:
        return sorted(files)

    prefer_ext = prefer_ext.lower().lstrip(".")
    by_stem = {}
    for f in files:
        stem, ext = os.path.splitext(f)
        ext = ext.lower().lstrip(".")
        by_stem.setdefault(stem, {})[ext] = f

    selected = []
    for stem, variants in by_stem.items():
        if prefer_ext in variants:
            selected.append(variants[prefer_ext])
        else:
            # Fall back to any available format (stable order by SLIDE_EXTENSIONS)
            for ext in (e.lstrip(".") for e in SLIDE_EXTENSIONS):
                if ext in variants:
                    selected.append(variants[ext])
                    break
    return sorted(selected)


def resolve_grid_output_base(
    slide_dir: str,
    output_base: str | None,
    patch_w: int,
    patch_h: int,
) -> str:
    """
    Root output folder for one --slide-dir batch.

    Default: <parent-of-slide-dir>/Tiles_{h}x{w}/{slide_folder_name}/
    Example: .../New Slides Eva/Tiles_1024x1024/EV P4 H and E stain/
    """
    slide_dir = os.path.expanduser(slide_dir.rstrip(os.sep))
    group_name = os.path.basename(slide_dir)
    if output_base is None or str(output_base).strip() == "":
        parent = os.path.dirname(slide_dir)
        output_base = os.path.join(parent, f"Tiles_{patch_h}x{patch_w}")
    else:
        output_base = os.path.expanduser(output_base)
    return os.path.join(output_base, group_name)


def slide_output_dir(group_output_base: str, slide_filename: str) -> str:
    """Per-slide folder: .../{slide_stem}/ with kept/ and discarded/."""
    stem = os.path.splitext(slide_filename)[0]
    return os.path.join(group_output_base, stem)


def run_grid(
    slide_dir,
    output_base,
    patch_w,
    patch_h,
    cfg: QualityConfig,
    overlap: float = 0.0,
    prefer_ext: str | None = "svs",
    save_rejected: bool = True,
    save_rejected_mask: bool = False,
):
    """Batch sliding-window generation over a folder of whole-slide images."""
    slide_dir = os.path.expanduser(slide_dir)
    group_output = resolve_grid_output_base(slide_dir, output_base, patch_w, patch_h)
    os.makedirs(group_output, exist_ok=True)
    files = list_slide_files(slide_dir, prefer_ext=prefer_ext)

    if not files:
        raise FileNotFoundError(
            f"No slide files {SLIDE_EXTENSIONS} found in {slide_dir}"
        )

    print(f"Output root: {group_output}")
    print(f"Layout: {{slide_stem}}/kept/  and  {{slide_stem}}/discarded/{{reason}}/")

    for file in tqdm(files, desc="Generating tiles"):
        full_path = os.path.join(slide_dir, file)
        output_dir = slide_output_dir(group_output, file)
        os.makedirs(output_dir, exist_ok=True)
        slide = pyvips.Image.new_from_file(full_path)

        tissue_mask = scale_x = scale_y = None
        if cfg.use_tissue_mask:
            tissue_mask, scale_x, scale_y = build_slide_tissue_mask(
                slide, cfg, file_path=full_path
            )
            if not np.any(tissue_mask):
                print(f"Warning: empty tissue mask for {file}; processing full grid.")
                tissue_mask = None

        saved = generate_tiles(
            width=slide.width,
            height=slide.height,
            patch_size_w=patch_w,
            patch_size_h=patch_h,
            output_dir=output_dir,
            slide=slide,
            cfg=cfg,
            overlap=overlap,
            tissue_mask=tissue_mask,
            scale_x=scale_x or 1.0,
            scale_y=scale_y or 1.0,
            save_rejected=save_rejected,
            save_rejected_mask=save_rejected_mask,
        )
        print(f"{file}: kept {saved} tiles -> {output_dir}")


def stitch_tiles_to_single_image(output_dir, width, height, patch_size_w, patch_size_h):
    """Reconstruct a full image from saved tile_*.png files in output_dir."""
    kept_dir = os.path.join(output_dir, "kept")
    if os.path.isdir(kept_dir):
        output_dir = kept_dir

    full_image = pyvips.Image.black(width, height)

    tile_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".png") and f.startswith("tile_")
    ]

    for tile_file in tile_files:
        parts = tile_file.split("_")
        x = int(parts[1])
        y = int(parts[2].split(".")[0])

        tile = pyvips.Image.new_from_file(os.path.join(output_dir, tile_file))
        full_image = full_image.insert(tile, x, y)

    output_path = os.path.join(output_dir, "reconstructed_image.png")
    full_image.pngsave(output_path)
    return output_path


def run_stitch(tile_dir, slide_path, patch_w, patch_h):
    tile_dir = os.path.expanduser(tile_dir)
    slide_path = os.path.expanduser(slide_path)
    slide = pyvips.Image.new_from_file(slide_path)
    output_path = stitch_tiles_to_single_image(
        output_dir=tile_dir,
        width=slide.width,
        height=slide.height,
        patch_size_w=patch_w,
        patch_size_h=patch_h,
    )
    print(f"Reconstructed image saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate or stitch whole-slide image tiles with tissue-aware filtering."
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "random", "stitch"],
        default="grid",
        help="grid: sliding-window over a slide folder; "
        "random: random tiles from one slide; "
        "stitch: rebuild one image from tiles (default: grid)",
    )

    # Shared geometry
    parser.add_argument("--patch-w", type=int, default=1637, help="Tile width (default: 1637)")
    parser.add_argument("--patch-h", type=int, default=1018, help="Tile height (default: 1018)")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Grid overlap fraction in [0, 0.9] (default: 0). Try 0.1–0.25 for more coverage.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tiles (random mode)",
    )

    # Legacy intensity filter (kept as a weak secondary check)
    parser.add_argument(
        "--lower-intensity",
        type=int,
        default=240,
        help="Intensity blank filter lower bound (default: 240)",
    )
    parser.add_argument(
        "--upper-intensity",
        type=int,
        default=255,
        help="Intensity blank filter upper bound (default: 255)",
    )
    parser.add_argument(
        "--no-intensity-filter",
        action="store_true",
        help="Disable mean-intensity blank filter",
    )

    # Tissue pixel / fraction
    parser.add_argument(
        "--sat-min",
        type=int,
        default=24,
        help="HSV saturation threshold for tissue pixels (default: 24)",
    )
    parser.add_argument(
        "--value-max",
        type=int,
        default=238,
        help="HSV value upper bound for tissue pixels (default: 238)",
    )
    parser.add_argument(
        "--min-tissue-frac",
        type=float,
        default=0.45,
        help="Minimum tissue pixel fraction per tile (default: 0.45)",
    )
    parser.add_argument(
        "--min-filled-tissue-frac",
        type=float,
        default=0.75,
        help="If raw tissue frac is low, keep when hole-filled tissue frac >= this "
        "(alveolar / distributed white; default: 0.75)",
    )
    parser.add_argument(
        "--max-border-bg-frac",
        type=float,
        default=0.20,
        help="Max border-connected white fraction allowed when recovering low-tissue "
        "tiles (contiguous blank; default: 0.20)",
    )
    parser.add_argument(
        "--min-sat-mean",
        type=float,
        default=15.0,
        help="Minimum mean HSV saturation; rejects pale haze/blank (default: 15)",
    )
    parser.add_argument(
        "--min-gray-std",
        type=float,
        default=30.0,
        help="Minimum grayscale std; rejects low-contrast haze (default: 30)",
    )
    parser.add_argument(
        "--min-edge-frac",
        type=float,
        default=0.008,
        help="Minimum Canny edge fraction; rejects structureless blur (default: 0.008)",
    )

    # Blur / pen
    parser.add_argument(
        "--min-blur-var",
        type=float,
        default=32.0,
        help="Minimum Laplacian variance; lower = blurrier reject (default: 32)",
    )
    parser.add_argument(
        "--no-blur-filter",
        action="store_true",
        help="Disable blur rejection",
    )
    parser.add_argument(
        "--pen-frac-max",
        type=float,
        default=0.08,
        help="Max allowed pen-mark pixel fraction (default: 0.08)",
    )
    parser.add_argument(
        "--no-pen-filter",
        action="store_true",
        help="Disable pen/marker rejection",
    )

    # Slide-level mask
    parser.add_argument(
        "--no-tissue-mask",
        action="store_true",
        help="Disable slide-level tissue mask prefilter",
    )
    parser.add_argument(
        "--mask-max-dim",
        type=int,
        default=2048,
        help="Max dimension of downsample used for tissue mask (default: 2048)",
    )
    parser.add_argument(
        "--mask-min-tile-frac",
        type=float,
        default=0.35,
        help="Min tissue fraction of tile bbox in slide mask (default: 0.35)",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Morphology kernel size for mask cleanup (default: 5)",
    )

    # Grid
    parser.add_argument(
        "--slide-dir",
        type=str,
        default="~/Documents/Data/ALI surgical/Control-healthy slides/",
        help="Folder of slides (.mrxs, .svs, ...) for grid mode",
    )
    parser.add_argument(
        "--prefer-ext",
        type=str,
        default="svs",
        help="When both .svs and .mrxs exist for a slide, prefer this extension "
        "(default: svs). Use 'none' to process all matching files.",
    )
    parser.add_argument(
        "--no-save-rejected",
        action="store_true",
        help="Flat output only (tile_*.png in slide folder). "
        "Default writes kept/ and discarded/{reason}/ for QC.",
    )
    parser.add_argument(
        "--save-rejected-mask",
        action="store_true",
        help="Also save mask-skipped tiles under discarded/mask_skip (large disk use)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Root for tile output. Default: <parent-of-slide-dir>/Tiles_{h}x{w}/",
    )

    # Random
    parser.add_argument("--image", type=str, default=None, help="Single slide path (random mode)")
    parser.add_argument(
        "--patch-type",
        choices=["square", "rectangle", "fixed"],
        default="fixed",
        help="Random patch shape (default: fixed)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Lower bound for random patch size (square/rectangle)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Upper bound for random patch size (square/rectangle)",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=200,
        help="Target number of saved random tiles (default: 200)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=800,
        help="Edge margin for uniform random fallback (default: 800)",
    )

    # Stitch
    parser.add_argument(
        "--tile-dir",
        type=str,
        default=None,
        help="Folder of tile_x_y.png files (stitch mode)",
    )
    parser.add_argument(
        "--slide",
        type=str,
        default=None,
        help="Original slide path for width/height (stitch mode)",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = quality_config_from_args(args)

    if args.mode == "grid":
        prefer = None if args.prefer_ext.lower() == "none" else args.prefer_ext
        run_grid(
            slide_dir=args.slide_dir,
            output_base=args.output_base,
            patch_w=args.patch_w,
            patch_h=args.patch_h,
            cfg=cfg,
            overlap=args.overlap,
            prefer_ext=prefer,
            save_rejected=not args.no_save_rejected,
            save_rejected_mask=args.save_rejected_mask,
        )
    elif args.mode == "random":
        if not args.image:
            parser.error("--image is required for --mode random")
        generate_random_tiles(
            image_path=args.image,
            patch_type=args.patch_type,
            num_tiles=args.num_tiles,
            cfg=cfg,
            output_dir=args.output_dir,
            patch_w=args.patch_w,
            patch_h=args.patch_h,
            min_size=args.min_size,
            max_size=args.max_size,
            margin=args.margin,
        )
    elif args.mode == "stitch":
        if not args.tile_dir or not args.slide:
            parser.error("--tile-dir and --slide are required for --mode stitch")
        run_stitch(
            tile_dir=args.tile_dir,
            slide_path=args.slide,
            patch_w=args.patch_w,
            patch_h=args.patch_h,
        )


if __name__ == "__main__":
    main()
