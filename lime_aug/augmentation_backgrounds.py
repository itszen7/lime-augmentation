import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from lime_aug.files import IMG_EXTS, get_label_path_for_image, list_images_in_dir, list_split_dirs
from lime_aug.superpixels import (
    SegmentationAlgorithm,
    apply_background_on_mask,
    get_segmentation_fn,
    load_and_resize_background,
    load_font,
    load_image_rgb,
    make_random_replacement_mask,
    save_superpixel_debug_image,
)
from lime_aug.yolo import bbox_mask_coverage, read_yolo_labels, write_yolo_labels, yolo_line_to_bbox_px


# -----------------------------
# Data / config
# -----------------------------
@dataclass
class Config:
    dataset_folder: Path
    backgrounds_folder: Path
    output_folder: Path

    num_permutations: int

    # NEW: defect coverage threshold (fraction 0..1)
    # if replaced pixels cover >= threshold of bbox area => remove label
    cover_threshold: float

    segmentation: str  # "quickshift" or "slic"
    replacement_prob: float
    superpixel_scale: float
    seed: int

    verbose: bool
    save_debug_superpixels: bool

    # NEW: always save permutation debug images with bboxes
    save_debug_perm_bboxes: bool


# -----------------------------
# Debug visualization
# -----------------------------
def save_perm_bbox_debug_image(
    perm_img_rgb: np.ndarray,
    decisions: List[Tuple[str, Tuple[float, float, float, float], float, bool]],
    out_path: Path,
    font_size: int = 18,
) -> None:
    """
    Save permutation image with bboxes drawn.
    decisions: list of (cls, bbox_xyxy, coverage, is_removed)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    im = Image.fromarray(perm_img_rgb)
    draw = ImageDraw.Draw(im)
    font = load_font(font_size)

    # Colors (kept vs removed)
    kept_color = (0, 255, 0)     # green
    removed_color = (255, 0, 0)  # red

    for cls, (x1, y1, x2, y2), cov, is_removed in decisions:
        color = removed_color if is_removed else kept_color
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        # bbox outline
        draw.rectangle([x1i, y1i, x2i, y2i], outline=color, width=3)

        # label text
        status = "REM" if is_removed else "KEEP"
        txt = f"{cls} {status} cov={cov:.2f}"

        tx, ty = x1i + 2, max(0, y1i - 22)
        # black outline for readability
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + dx, ty + dy), txt, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), txt, fill=color, font=font)

    im.save(out_path)


# -----------------------------
# Main processing
# -----------------------------
def process_split(
    split_dir: Path,
    cfg: Config,
    bg_paths: List[Path],
    segmentation_fn: SegmentationAlgorithm,
    rng: random.Random,
) -> None:
    images = list_images_in_dir(split_dir)

    # YOLO output structure
    out_img_split_dir = cfg.output_folder / "images" / split_dir.name
    out_lbl_split_dir = cfg.output_folder / "labels" / split_dir.name
    out_img_split_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_split_dir.mkdir(parents=True, exist_ok=True)

    # Debug folders
    dbg_sp_dir = cfg.output_folder / "debug_superpixels" / split_dir.name
    dbg_bb_dir = cfg.output_folder / "debug_perm_bboxes" / split_dir.name

    # mandatory: ALL backgrounds
    bg_paths_sorted = sorted(bg_paths, key=lambda p: p.name.lower())

    print(f"\n=== Split: {split_dir.name} ===", flush=True)
    print(f"Found {len(images)} images", flush=True)
    print(f"Backgrounds: {len(bg_paths_sorted)} (MANDATORY: ALL backgrounds)", flush=True)
    print(f"Total outputs per image: {len(bg_paths_sorted)} * {cfg.num_permutations}", flush=True)
    print(f"Cover threshold (remove if covered >=): {cfg.cover_threshold:.2f}", flush=True)

    for img_path in tqdm(images, desc=f"Processing {split_dir.name}", unit="img", disable=cfg.verbose):
        label_path = get_label_path_for_image(img_path)
        yolo_lines = read_yolo_labels(label_path)

        img = load_image_rgb(img_path)
        H, W = img.shape[:2]

        segments = segmentation_fn(img)

        # optional: one superpixel debug per original image
        if cfg.save_debug_superpixels:
            dbg_path = dbg_sp_dir / f"{img_path.stem}_superpixels.png"
            save_superpixel_debug_image(img, segments, dbg_path, font_size=18)

        stem = img_path.stem
        ext = img_path.suffix.lower()

        for bg_i, bg_path in enumerate(bg_paths_sorted):
            bg = load_and_resize_background(bg_path, (W, H))

            for k in range(cfg.num_permutations):
                replaced_mask = make_random_replacement_mask(segments, cfg.replacement_prob, rng)
                replaced_pct = 100.0 * float(replaced_mask.mean())

                perm_img = apply_background_on_mask(img, bg, replaced_mask)

                kept_lines: List[str] = []
                removed_count = 0

                # Track decisions for bbox-debug image
                decisions: List[Tuple[str, Tuple[float, float, float, float], float, bool]] = []

                for line in yolo_lines:
                    cls, bbox = yolo_line_to_bbox_px(line, W, H)
                    cov = bbox_mask_coverage(replaced_mask, bbox)

                    is_removed = cov >= cfg.cover_threshold
                    decisions.append((cls, bbox, cov, is_removed))

                    if cfg.verbose:
                        print(
                            f"    bg {bg_i:03d} ({bg_path.name}) | perm {k:03d} | "
                            f"cov={cov:.3f} | {'REMOVE' if is_removed else 'KEEP'} | {line}",
                            flush=True,
                        )

                    if is_removed:
                        removed_count += 1
                    else:
                        kept_lines.append(line)

                out_stem = f"{stem}_bg{bg_i:03d}_perm{k:03d}"

                # Save image
                out_img_path = out_img_split_dir / f"{out_stem}{ext}"
                Image.fromarray(perm_img).save(out_img_path)

                # Save label txt (ALWAYS, even if empty)
                out_lbl_path = out_lbl_split_dir / f"{out_stem}.txt"
                write_yolo_labels(out_lbl_path, kept_lines)

                # Save debug permutation image with bboxes
                if cfg.save_debug_perm_bboxes:
                    dbg_perm_path = dbg_bb_dir / f"{out_stem}.png"
                    save_perm_bbox_debug_image(perm_img, decisions, dbg_perm_path, font_size=18)

                if not cfg.verbose:
                    print(
                        f"[{split_dir.name}] {img_path.name} -> {out_stem} | "
                        f"bg={bg_path.name} | replaced={replaced_pct:.1f}% | "
                        f"labels {len(yolo_lines)} -> {len(kept_lines)} (removed {removed_count})",
                        flush=True,
                    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate LIME-style superpixel permutations with background replacements.\n"
            "MANDATORY: generates permutations for ALL backgrounds for every image.\n"
            "Label removal rule: if replaced mask covers >= X% of bbox area, remove that label.\n"
            "Also saves debug images for each permutation with bboxes drawn."
        )
    )
    ap.add_argument("--dataset_folder", type=str, required=True)
    ap.add_argument("--backgrounds_folder", type=str, required=True)
    ap.add_argument("--output_folder", type=str, required=True)

    ap.add_argument("--num_permutations", type=int, default=5, help="Permutations PER BACKGROUND per image")

    # allow older --iou_threshold usage but map to cover_threshold
    ap.add_argument(
        "--cover_threshold",
        "--iou_threshold",
        dest="cover_threshold",
        type=float,
        default=0.5,
        help="Remove label if replaced mask covers >= this fraction of the bbox area (0..1).",
    )

    ap.add_argument("--segmentation", type=str, default="quickshift", choices=["quickshift", "slic"])
    ap.add_argument("--replacement_prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--superpixel_scale", type=float, default=1.0)

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save_debug_superpixels", action="store_true")

    args = ap.parse_args()

    cfg = Config(
        dataset_folder=Path(args.dataset_folder),
        backgrounds_folder=Path(args.backgrounds_folder),
        output_folder=Path(args.output_folder),
        num_permutations=args.num_permutations,
        cover_threshold=float(args.cover_threshold),
        segmentation=args.segmentation,
        replacement_prob=args.replacement_prob,
        superpixel_scale=args.superpixel_scale,
        seed=args.seed,
        verbose=args.verbose,
        save_debug_superpixels=args.save_debug_superpixels,
        save_debug_perm_bboxes=True,  # requested: always create debug folder with bboxes
    )

    if not cfg.dataset_folder.exists():
        print(f"ERROR: dataset_folder does not exist: {cfg.dataset_folder}", flush=True)
        sys.exit(1)
    if not cfg.backgrounds_folder.exists():
        print(f"ERROR: backgrounds_folder does not exist: {cfg.backgrounds_folder}", flush=True)
        sys.exit(1)

    bg_paths = [p for p in cfg.backgrounds_folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not bg_paths:
        print(f"ERROR: No background images found in {cfg.backgrounds_folder}", flush=True)
        sys.exit(1)

    if not (0.0 <= cfg.cover_threshold <= 1.0):
        print("ERROR: cover_threshold must be between 0 and 1.", flush=True)
        sys.exit(1)

    print("=== Config ===", flush=True)
    print(f"dataset_folder       : {cfg.dataset_folder}", flush=True)
    print(f"backgrounds_folder   : {cfg.backgrounds_folder} ({len(bg_paths)} images)", flush=True)
    print(f"output_folder        : {cfg.output_folder}", flush=True)
    print(f"num_permutations/bg  : {cfg.num_permutations}", flush=True)
    print(f"TOTAL perms per image: {len(bg_paths)} * {cfg.num_permutations} = {len(bg_paths) * cfg.num_permutations}", flush=True)
    print(f"cover_threshold      : {cfg.cover_threshold}", flush=True)
    print(f"segmentation         : {cfg.segmentation}", flush=True)
    print(f"replacement_prob     : {cfg.replacement_prob}", flush=True)
    print(f"superpixel_scale     : {cfg.superpixel_scale}", flush=True)
    print(f"seed                 : {cfg.seed}", flush=True)
    print(f"verbose              : {cfg.verbose}", flush=True)
    print(f"save_debug_superpixels: {cfg.save_debug_superpixels}", flush=True)
    print("==============\n", flush=True)

    rng = random.Random(cfg.seed)
    segmentation_fn = get_segmentation_fn(cfg.segmentation, cfg.superpixel_scale)

    split_dirs = list_split_dirs(cfg.dataset_folder)
    for split_dir in split_dirs:
        process_split(split_dir, cfg, bg_paths, segmentation_fn, rng)

    print("\nDone. Output written to:", cfg.output_folder, flush=True)
    print("Debug folders:", flush=True)
    print(f" - superpixels: {cfg.output_folder / 'debug_superpixels'}", flush=True)
    print(f" - perm bboxes: {cfg.output_folder / 'debug_perm_bboxes'}", flush=True)


if __name__ == "__main__":
    main()
