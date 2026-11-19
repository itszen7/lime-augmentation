#!/usr/bin/env python3
"""
Generate LIME-style superpixel permutations from a plain image folder (NO labels).

Given:
  --images_folder   (folder containing only images, optionally in subfolders)
Creates:
  --output_folder   (folder containing permutations)

Permutation idea:
  - segment image into superpixels (quickshift or slic)
  - for each permutation, randomly pick superpixels to "replace"
  - replaced pixels are filled using a chosen fill mode:
      mean | blur | noise | black | white

Outputs:
  - <output_folder>/(optional subdirs)/<stem>_perm000.<ext>
  - optional debug superpixel visualization images (boundaries + ids)
"""

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from lime_aug.files import IMG_EXTS
from lime_aug.superpixels import (
    SegmentationAlgorithm,
    get_segmentation_fn,
    load_image_rgb,
    make_random_replacement_mask,
    save_superpixel_debug_image,
)


@dataclass
class Config:
    images_folder: Path
    output_folder: Path

    num_permutations: int
    segmentation: str  # "quickshift" or "slic"
    replacement_prob: float
    superpixel_scale: float
    seed: int

    recursive: bool
    preserve_structure: bool

    fill_mode: str  # mean | blur | noise | black | white
    blur_radius: float

    save_debug_superpixels: bool
    verbose: bool


# -----------------------------
# File helpers
# -----------------------------
def list_images(folder: Path, recursive: bool) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    files.sort()
    return files


def rel_subdir(root: Path, img_path: Path) -> Path:
    """Return subdir relative to root, excluding filename."""
    try:
        r = img_path.relative_to(root)
        return r.parent
    except Exception:
        return Path("")


# -----------------------------
# Image / segmentation helpers
# -----------------------------
def apply_fill_on_mask(
    image_rgb: np.ndarray,
    replaced_mask: np.ndarray,
    fill_mode: str,
    rng: random.Random,
    blur_radius: float,
) -> np.ndarray:
    """Replace masked pixels using the chosen fill mode."""
    fill_mode = fill_mode.lower().strip()
    out = image_rgb.copy()

    if fill_mode == "mean":
        mean_color = image_rgb.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        out[replaced_mask] = mean_color
        return out

    if fill_mode == "black":
        out[replaced_mask] = (0, 0, 0)
        return out

    if fill_mode == "white":
        out[replaced_mask] = (255, 255, 255)
        return out

    if fill_mode == "noise":
        H, W = replaced_mask.shape
        noise = np.empty((H, W, 3), dtype=np.uint8)
        # deterministic-ish noise based on rng
        noise[..., 0] = np.array([rng.randrange(256) for _ in range(H * W)], dtype=np.uint8).reshape(H, W)
        noise[..., 1] = np.array([rng.randrange(256) for _ in range(H * W)], dtype=np.uint8).reshape(H, W)
        noise[..., 2] = np.array([rng.randrange(256) for _ in range(H * W)], dtype=np.uint8).reshape(H, W)
        out[replaced_mask] = noise[replaced_mask]
        return out

    if fill_mode == "blur":
        # blur whole image and paste blurred pixels where mask is true
        im = Image.fromarray(image_rgb)
        blurred = im.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        blurred_rgb = np.array(blurred)
        out[replaced_mask] = blurred_rgb[replaced_mask]
        return out

    raise ValueError("fill_mode must be one of: mean, blur, noise, black, white")


# -----------------------------
# Main processing
# -----------------------------
def process_one_image(
    img_path: Path,
    cfg: Config,
    segmentation_fn: SegmentationAlgorithm,
    rng: random.Random,
) -> None:
    img = load_image_rgb(img_path)
    H, W = img.shape[:2]

    segments = segmentation_fn(img)

    # Determine output subfolder strategy
    subdir = rel_subdir(cfg.images_folder, img_path) if (cfg.preserve_structure and cfg.recursive) else Path("")
    out_dir = cfg.output_folder / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_debug_superpixels:
        dbg_dir = cfg.output_folder / "_debug_superpixels" / subdir
        dbg_dir.mkdir(parents=True, exist_ok=True)
        dbg_path = dbg_dir / f"{img_path.stem}_superpixels.png"
        save_superpixel_debug_image(img, segments, dbg_path, font_size=18)

    ext = img_path.suffix.lower()
    stem = img_path.stem

    for k in range(cfg.num_permutations):
        replaced_mask = make_random_replacement_mask(segments, cfg.replacement_prob, rng)
        replaced_pct = 100.0 * float(replaced_mask.mean())

        perm_img = apply_fill_on_mask(
            image_rgb=img,
            replaced_mask=replaced_mask,
            fill_mode=cfg.fill_mode,
            rng=rng,
            blur_radius=cfg.blur_radius,
        )

        out_name = f"{stem}_perm{k:03d}{ext}"
        out_path = out_dir / out_name
        Image.fromarray(perm_img).save(out_path)

        if cfg.verbose:
            print(f"{img_path.name} -> {out_path} | replaced={replaced_pct:.1f}%", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate LIME-style superpixel permutations from an image folder (no labels).\n"
            "Each permutation replaces a random subset of superpixels using a chosen fill mode."
        )
    )
    ap.add_argument("--images_folder", type=str, required=True, help="Folder with images.")
    ap.add_argument("--output_folder", type=str, required=True, help="Folder to write permutations into.")

    ap.add_argument("--num_permutations", type=int, default=10, help="Permutations per image.")

    ap.add_argument("--segmentation", type=str, default="quickshift", choices=["quickshift", "slic"])
    ap.add_argument("--replacement_prob", type=float, default=0.5, help="Probability of replacing each superpixel.")
    ap.add_argument("--superpixel_scale", type=float, default=1.0, help="Scale controlling superpixel granularity.")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--recursive", action="store_true", help="If set, scan images_folder recursively.")
    ap.add_argument(
        "--preserve_structure",
        action="store_true",
        help="If set (and --recursive), preserves subfolder structure under output_folder.",
    )

    ap.add_argument(
        "--fill_mode",
        type=str,
        default="mean",
        choices=["mean", "blur", "noise", "black", "white"],
        help="How to fill replaced superpixels.",
    )
    ap.add_argument("--blur_radius", type=float, default=8.0, help="Gaussian blur radius (only for fill_mode=blur).")

    ap.add_argument("--save_debug_superpixels", action="store_true", help="Save a superpixel boundary/id debug image.")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    cfg = Config(
        images_folder=Path(args.images_folder),
        output_folder=Path(args.output_folder),
        num_permutations=int(args.num_permutations),
        segmentation=str(args.segmentation),
        replacement_prob=float(args.replacement_prob),
        superpixel_scale=float(args.superpixel_scale),
        seed=int(args.seed),
        recursive=bool(args.recursive),
        preserve_structure=bool(args.preserve_structure),
        fill_mode=str(args.fill_mode),
        blur_radius=float(args.blur_radius),
        save_debug_superpixels=bool(args.save_debug_superpixels),
        verbose=bool(args.verbose),
    )

    if not cfg.images_folder.exists():
        print(f"ERROR: images_folder does not exist: {cfg.images_folder}", flush=True)
        sys.exit(1)

    if not (0.0 < cfg.replacement_prob <= 1.0):
        print("ERROR: replacement_prob must be in (0, 1].", flush=True)
        sys.exit(1)

    cfg.output_folder.mkdir(parents=True, exist_ok=True)

    images = list_images(cfg.images_folder, cfg.recursive)
    if not images:
        print(f"ERROR: No images found under: {cfg.images_folder}", flush=True)
        sys.exit(1)

    print("=== Config ===", flush=True)
    print(f"images_folder      : {cfg.images_folder}", flush=True)
    print(f"output_folder      : {cfg.output_folder}", flush=True)
    print(f"num_permutations   : {cfg.num_permutations}", flush=True)
    print(f"segmentation       : {cfg.segmentation}", flush=True)
    print(f"replacement_prob   : {cfg.replacement_prob}", flush=True)
    print(f"superpixel_scale   : {cfg.superpixel_scale}", flush=True)
    print(f"seed               : {cfg.seed}", flush=True)
    print(f"recursive          : {cfg.recursive}", flush=True)
    print(f"preserve_structure : {cfg.preserve_structure}", flush=True)
    print(f"fill_mode          : {cfg.fill_mode}", flush=True)
    if cfg.fill_mode == "blur":
        print(f"blur_radius        : {cfg.blur_radius}", flush=True)
    print(f"save_debug_superpixels : {cfg.save_debug_superpixels}", flush=True)
    print(f"images found       : {len(images)}", flush=True)
    print("==============\n", flush=True)

    rng = random.Random(cfg.seed)
    segmentation_fn = get_segmentation_fn(cfg.segmentation, cfg.superpixel_scale)

    for img_path in tqdm(images, desc="Processing images", unit="img", disable=cfg.verbose):
        process_one_image(img_path, cfg, segmentation_fn, rng)

    print("\nDone. Output written to:", cfg.output_folder, flush=True)
    if cfg.save_debug_superpixels:
        print("Debug superpixels written to:", cfg.output_folder / "_debug_superpixels", flush=True)


if __name__ == "__main__":
    main()
