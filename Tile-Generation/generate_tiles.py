"""
Tile generation for whole-slide images (.mrxs).

Converted from generate_tiles.ipynb.
Supports random tile sampling, fixed sliding-window tiling, and stitch reconstruction.
"""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
import pyvips
from tqdm import tqdm


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
    return np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=np.uint8,
        shape=[vips_image.height, vips_image.width, vips_image.bands],
    )


def process_tile(
    slide,
    width,
    height,
    x,
    y,
    patch_size_w,
    patch_size_h,
    lower_bnd_intensity,
    upper_bnd_intensity,
    fdir,
):
    actual_patch_w = min(patch_size_w, width - x)
    actual_patch_h = min(patch_size_h, height - y)

    tile = slide.crop(x, y, actual_patch_w, actual_patch_h)
    tile_array = pyvips_to_numpy(tile)
    mean_value = np.mean(tile_array)

    if lower_bnd_intensity < mean_value <= upper_bnd_intensity:
        return None

    output_filename = f"{fdir}/tile_{x}_{y}.jpg"
    tile.write_to_file(output_filename)
    return output_filename


def process_tile_2(
    x,
    y,
    width,
    height,
    patch_size_w,
    patch_size_h,
    lower_bnd_intensity,
    upper_bnd_intensity,
    file_path,
    output_dir,
    slide,
):
    actual_patch_w = min(patch_size_w, width - x)
    actual_patch_h = min(patch_size_h, height - y)

    tile = slide.crop(x, y, actual_patch_w, actual_patch_h)
    tile_array = pyvips_to_numpy(tile)
    mean_value = np.mean(tile_array)

    if lower_bnd_intensity < mean_value <= upper_bnd_intensity:
        return None

    output_filename = os.path.expanduser(f"{output_dir}/tile_{x}_{y}.png")
    tile.pngsave(output_filename, compression=9)
    return output_filename


# ---------------------------------------------------------------------------
# Random tile generation
# ---------------------------------------------------------------------------

def generate_random_tiles(
    image_path,
    patch_type,
    num_tiles,
    lower_bnd_intensity,
    upper_bnd_intensity,
    output_dir=None,
    patch_w=None,
    patch_h=None,
    min_size=None,
    max_size=None,
    margin=800,
):
    """Generate random tiles from a single slide."""
    image_path = os.path.expanduser(image_path)
    slide = pyvips.Image.new_from_file(image_path)
    width = slide.width
    height = slide.height

    if width <= margin or height <= margin:
        raise ValueError(
            f"Slide ({width}x{height}) is smaller than margin={margin}; "
            "reduce --margin or use a larger slide."
        )

    if patch_type == "fixed":
        if patch_w is None or patch_h is None:
            raise ValueError("--patch-w and --patch-h are required for patch-type=fixed")
        fixed_w, fixed_h = patch_w, patch_h
        default_dir = f"Tile{fixed_h}{fixed_w}_{image_path[-6:-5]}"
    else:
        if min_size is None or max_size is None:
            raise ValueError(
                "--min-size and --max-size are required for patch-type="
                f"{patch_type}"
            )
        if min_size > max_size:
            raise ValueError("--min-size must be <= --max-size")
        default_dir = f"Tile_{min_size}_{max_size}_{image_path[-6:-5]}"

    fdir = os.path.expanduser(output_dir) if output_dir else default_dir
    os.makedirs(fdir, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = []

        for _ in range(num_tiles):
            x = random.randint(0, width - margin)
            y = random.randint(0, height - margin)

            if patch_type == "square":
                patch_size_w = patch_size_h = random.randint(min_size, max_size)
            elif patch_type == "rectangle":
                patch_size_w = random.randint(min_size, max_size)
                patch_size_h = random.randint(min_size, max_size)
            elif patch_type == "fixed":
                patch_size_w = fixed_w
                patch_size_h = fixed_h
            else:
                raise ValueError(f"Unknown patch type: {patch_type}")

            futures.append(
                executor.submit(
                    process_tile,
                    slide,
                    width,
                    height,
                    x,
                    y,
                    patch_size_w,
                    patch_size_h,
                    lower_bnd_intensity,
                    upper_bnd_intensity,
                    fdir,
                )
            )

        for future in futures:
            result = future.result()
            if result:
                print(f"Saved: {result}")


# ---------------------------------------------------------------------------
# Fixed sliding-window tile generation
# ---------------------------------------------------------------------------

def generate_tiles(
    width,
    height,
    patch_size_w,
    patch_size_h,
    lower_bnd_intensity,
    upper_bnd_intensity,
    file_path,
    output_dir,
    slide,
):
    """Slide a fixed tile window across the entire slide and save non-blank tiles."""
    with ThreadPoolExecutor() as executor:
        futures = []
        for y in range(0, height, patch_size_h):
            for x in range(0, width, patch_size_w):
                futures.append(
                    executor.submit(
                        process_tile_2,
                        x,
                        y,
                        width,
                        height,
                        patch_size_w,
                        patch_size_h,
                        lower_bnd_intensity,
                        upper_bnd_intensity,
                        file_path,
                        output_dir,
                        slide,
                    )
                )

        for future in futures:
            future.result()


def run_grid(
    slide_dir,
    output_base,
    patch_w,
    patch_h,
    lower_bnd_intensity,
    upper_bnd_intensity,
):
    """Batch sliding-window generation over a folder of .mrxs slides."""
    slide_dir = os.path.expanduser(slide_dir)
    output_base = os.path.expanduser(output_base)
    files = [f for f in os.listdir(slide_dir) if f.endswith(".mrxs")]

    if not files:
        raise FileNotFoundError(f"No .mrxs files found in {slide_dir}")

    for file in tqdm(files, desc="Generating Tiles .mrxs files"):
        full_path = os.path.join(slide_dir, file)
        output_dir = os.path.join(
            output_base, f"Tiles_{patch_h}_{patch_w}_{file}"
        )
        os.makedirs(output_dir, exist_ok=True)
        slide = pyvips.Image.new_from_file(full_path)

        generate_tiles(
            width=slide.width,
            height=slide.height,
            patch_size_w=patch_w,
            patch_size_h=patch_h,
            lower_bnd_intensity=lower_bnd_intensity,
            upper_bnd_intensity=upper_bnd_intensity,
            file_path=full_path,
            output_dir=output_dir,
            slide=slide,
        )


def stitch_tiles_to_single_image(output_dir, width, height, patch_size_w, patch_size_h):
    """Reconstruct a full image from saved tile_*.png files in output_dir."""
    full_image = pyvips.Image.black(width, height)

    tile_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]

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
        description="Generate or stitch whole-slide image tiles."
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "random", "stitch"],
        default="grid",
        help="grid: sliding-window over a slide folder; "
        "random: random tiles from one slide; "
        "stitch: rebuild one image from tiles (default: grid)",
    )

    # Shared
    parser.add_argument("--patch-w", type=int, default=1637, help="Tile width (default: 1637)")
    parser.add_argument("--patch-h", type=int, default=1018, help="Tile height (default: 1018)")
    parser.add_argument(
        "--lower-intensity",
        type=int,
        default=240,
        help="Skip tiles with mean intensity above this (default: 240)",
    )
    parser.add_argument(
        "--upper-intensity",
        type=int,
        default=255,
        help="Skip tiles with mean intensity <= this (default: 255)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tiles (random mode) or override path",
    )

    # Grid
    parser.add_argument(
        "--slide-dir",
        type=str,
        default="~/Documents/Data/ALI surgical/Control-healthy slides/",
        help="Folder of .mrxs slides (grid mode)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="~/Documents/Code/Lung_Injury/Healthy/",
        help="Parent directory for per-slide tile folders (grid mode)",
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
        help="Number of random tiles to attempt (default: 200)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=800,
        help="Keep random crop origin this many pixels from edges (default: 800)",
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

    if args.mode == "grid":
        run_grid(
            slide_dir=args.slide_dir,
            output_base=args.output_base,
            patch_w=args.patch_w,
            patch_h=args.patch_h,
            lower_bnd_intensity=args.lower_intensity,
            upper_bnd_intensity=args.upper_intensity,
        )
    elif args.mode == "random":
        if not args.image:
            parser.error("--image is required for --mode random")
        generate_random_tiles(
            image_path=args.image,
            patch_type=args.patch_type,
            num_tiles=args.num_tiles,
            lower_bnd_intensity=args.lower_intensity,
            upper_bnd_intensity=args.upper_intensity,
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
