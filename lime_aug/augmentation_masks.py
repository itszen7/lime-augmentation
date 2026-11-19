import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from skimage.measure import label as cc_label

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

    # Default cover threshold for all classes
    cover_threshold_default: float

    # Special threshold for "trama" class (match by class token in YOLO txt)
    trama_class: str
    cover_threshold_trama: float

    segmentation: str  # "quickshift" or "slic"
    replacement_prob: float
    superpixel_scale: float
    seed: int

    verbose: bool
    save_debug_superpixels: bool

    # Always save permutation debug images with bboxes
    save_debug_perm_bboxes: bool

    # NEW: keep originals in the augmented dataset
    keep_originals: bool

    # Splitting controls when cov > 0 (mask touches bbox)
    min_component_pixels: int
    min_component_w_px: int
    min_component_h_px: int
    component_pad_px: int
    connectivity: int  # 1 => 4-connectivity, 2 => 8-connectivity


# -----------------------------
# YOLO label helpers
# -----------------------------
def bbox_xyxy_to_yolo_line(cls: str, bbox_xyxy: Tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    """Convert pixel bbox (x1,y1,x2,y2) to YOLO normalized line: cls xc yc w h"""
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        bw = max(bw, 1.0)
        bh = max(bh, 1.0)

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    xcn = xc / img_w
    ycn = yc / img_h
    bwn = bw / img_w
    bhn = bh / img_h

    xcn = max(0.0, min(1.0, xcn))
    ycn = max(0.0, min(1.0, ycn))
    bwn = max(0.0, min(1.0, bwn))
    bhn = max(0.0, min(1.0, bhn))

    return f"{cls} {xcn:.6f} {ycn:.6f} {bwn:.6f} {bhn:.6f}"


# -----------------------------
# Split bbox into multiple bboxes by visible components
# -----------------------------
def split_bbox_by_visible_components(
    replaced_mask: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    min_component_pixels: int,
    min_w_px: int,
    min_h_px: int,
    pad_px: int,
    connectivity: int,
) -> List[Tuple[float, float, float, float]]:
    """
    If cov>0, compute connected components of visible pixels inside bbox:
        visible = ~replaced_mask
    Return one bbox per connected component.
    """
    x1, y1, x2, y2 = bbox_xyxy
    H, W = replaced_mask.shape

    x1i = max(0, min(W, int(np.floor(x1))))
    y1i = max(0, min(H, int(np.floor(y1))))
    x2i = max(0, min(W, int(np.ceil(x2))))
    y2i = max(0, min(H, int(np.ceil(y2))))

    if x2i <= x1i or y2i <= y1i:
        return []

    region_replaced = replaced_mask[y1i:y2i, x1i:x2i]
    region_visible = ~region_replaced

    lab = cc_label(region_visible.astype(np.uint8), connectivity=connectivity)
    n = int(lab.max())
    if n <= 0:
        return []

    out: List[Tuple[float, float, float, float]] = []
    for comp_id in range(1, n + 1):
        ys, xs = np.where(lab == comp_id)
        if xs.size == 0:
            continue

        area = int(xs.size)
        if area < int(min_component_pixels):
            continue

        rx1 = int(xs.min())
        rx2 = int(xs.max()) + 1
        ry1 = int(ys.min())
        ry2 = int(ys.max()) + 1

        bw = rx2 - rx1
        bh = ry2 - ry1
        if bw < int(min_w_px) or bh < int(min_h_px):
            continue

        nx1 = x1i + rx1 - int(pad_px)
        nx2 = x1i + rx2 + int(pad_px)
        ny1 = y1i + ry1 - int(pad_px)
        ny2 = y1i + ry2 + int(pad_px)

        nx1 = max(0, min(W, nx1))
        nx2 = max(0, min(W, nx2))
        ny1 = max(0, min(H, ny1))
        ny2 = max(0, min(H, ny2))

        if nx2 > nx1 and ny2 > ny1:
            out.append((float(nx1), float(ny1), float(nx2), float(ny2)))

    out.sort(key=lambda b: (b[0], b[1]))
    return out


# -----------------------------
# Debug visualization
# -----------------------------
def save_perm_bbox_debug_image(
    perm_img_rgb: np.ndarray,
    decisions: List[Tuple[str, Tuple[float, float, float, float], float, float, bool]],
    out_path: Path,
    font_size: int = 18,
) -> None:
    """
    decisions: list of (cls, bbox_xyxy, coverage, threshold_used, is_removed)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    im = Image.fromarray(perm_img_rgb)
    draw = ImageDraw.Draw(im)
    font = load_font(font_size)

    kept_color = (0, 255, 0)
    removed_color = (255, 0, 0)

    for cls, (x1, y1, x2, y2), cov, thr, is_removed in decisions:
        color = removed_color if is_removed else kept_color
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        draw.rectangle([x1i, y1i, x2i, y2i], outline=color, width=3)

        status = "REM" if is_removed else "KEEP"
        txt = f"{cls} {status} cov={cov:.2f} thr={thr:.2f}"

        tx, ty = x1i + 2, max(0, y1i - 22)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + dx, ty + dy), txt, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), txt, fill=color, font=font)

    im.save(out_path)


# -----------------------------
# Dataset helpers
# -----------------------------
def threshold_for_class(cfg: Config, cls_token: str) -> float:
    """Return cover threshold for given class token."""
    if cls_token == cfg.trama_class:
        return cfg.cover_threshold_trama
    return cfg.cover_threshold_default


def copy_original_pair_to_output(cfg: Config, split_name: str, img_path: Path) -> None:
    """
    Copy the original image and its label (or empty label if missing) to output dataset.
    Keeps the exact same file name.
    """
    out_img = cfg.output_folder / "images" / split_name / img_path.name
    out_lbl = cfg.output_folder / "labels" / split_name / f"{img_path.stem}.txt"

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    # Copy image bytes losslessly
    out_img.write_bytes(img_path.read_bytes())

    # Copy label or create empty
    src_lbl = get_label_path_for_image(img_path)
    if src_lbl.exists():
        out_lbl.write_text(src_lbl.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        out_lbl.write_text("", encoding="utf-8")


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

    out_img_split_dir = cfg.output_folder / "images" / split_dir.name
    out_lbl_split_dir = cfg.output_folder / "labels" / split_dir.name
    out_img_split_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_split_dir.mkdir(parents=True, exist_ok=True)

    dbg_sp_dir = cfg.output_folder / "debug_superpixels" / split_dir.name
    dbg_bb_dir = cfg.output_folder / "debug_perm_bboxes" / split_dir.name

    bg_paths_sorted = sorted(bg_paths, key=lambda p: p.name.lower())

    print(f"\n=== Split: {split_dir.name} ===", flush=True)
    print(f"Found {len(images)} images", flush=True)
    print(f"Backgrounds: {len(bg_paths_sorted)} (MANDATORY: ALL backgrounds)", flush=True)
    print(f"Total outputs per image: {len(bg_paths_sorted)} * {cfg.num_permutations}", flush=True)
    print(
        f"Thresholds: default={cfg.cover_threshold_default:.2f} | "
        f"trama(class={cfg.trama_class})={cfg.cover_threshold_trama:.2f}",
        flush=True,
    )
    print(
        f"Split settings: min_pixels={cfg.min_component_pixels}, "
        f"min_w={cfg.min_component_w_px}, min_h={cfg.min_component_h_px}, "
        f"pad={cfg.component_pad_px}, conn={cfg.connectivity}",
        flush=True,
    )
    print(f"Keep originals: {cfg.keep_originals}", flush=True)

    for img_path in tqdm(images, desc=f"Processing {split_dir.name}", unit="img", disable=cfg.verbose):
        label_path = get_label_path_for_image(img_path)
        yolo_lines = read_yolo_labels(label_path)

        # NEW: keep originals in output dataset
        if cfg.keep_originals:
            copy_original_pair_to_output(cfg, split_dir.name, img_path)

        img = load_image_rgb(img_path)
        H, W = img.shape[:2]

        segments = segmentation_fn(img)

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

                decisions: List[Tuple[str, Tuple[float, float, float, float], float, float, bool]] = []

                for line in yolo_lines:
                    cls, bbox = yolo_line_to_bbox_px(line, W, H)
                    cov = bbox_mask_coverage(replaced_mask, bbox)
                    thr = threshold_for_class(cfg, cls)

                    if cov >= thr:
                        removed_count += 1
                        decisions.append((cls, bbox, cov, thr, True))
                        if cfg.verbose:
                            print(f"REMOVE cls={cls} cov={cov:.3f} >= thr={thr:.3f} | {line}", flush=True)
                        continue

                    if cov > 0.0:
                        boxes = split_bbox_by_visible_components(
                            replaced_mask=replaced_mask,
                            bbox_xyxy=bbox,
                            min_component_pixels=cfg.min_component_pixels,
                            min_w_px=cfg.min_component_w_px,
                            min_h_px=cfg.min_component_h_px,
                            pad_px=cfg.component_pad_px,
                            connectivity=cfg.connectivity,
                        )

                        if not boxes:
                            removed_count += 1
                            decisions.append((cls, bbox, cov, thr, True))
                            if cfg.verbose:
                                print(f"REMOVE cls={cls} (cov>0 but no comps) | {line}", flush=True)
                            continue

                        for b in boxes:
                            kept_lines.append(bbox_xyxy_to_yolo_line(cls, b, W, H))
                            decisions.append((cls, b, cov, thr, False))

                        if cfg.verbose:
                            print(f"SPLIT cls={cls} cov={cov:.3f} -> {len(boxes)} boxes | {line}", flush=True)
                        continue

                    kept_lines.append(line)
                    decisions.append((cls, bbox, cov, thr, False))
                    if cfg.verbose:
                        print(f"KEEP cls={cls} cov=0 thr={thr:.3f} | {line}", flush=True)

                out_stem = f"{stem}_bg{bg_i:03d}_perm{k:03d}"

                out_img_path = out_img_split_dir / f"{out_stem}{ext}"
                Image.fromarray(perm_img).save(out_img_path)

                out_lbl_path = out_lbl_split_dir / f"{out_stem}.txt"
                write_yolo_labels(out_lbl_path, kept_lines)

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
            "Label rules:\n"
            " - remove if replaced mask covers >= per-class threshold of bbox area\n"
            " - if cov > 0 and not removed, split bbox into multiple bboxes via visible components\n"
            "Also saves debug images for each permutation with bboxes drawn."
        )
    )
    ap.add_argument("--dataset_folder", type=str, required=True)
    ap.add_argument("--backgrounds_folder", type=str, required=True)
    ap.add_argument("--output_folder", type=str, required=True)

    ap.add_argument("--num_permutations", type=int, default=5, help="Permutations PER BACKGROUND per image")

    ap.add_argument(
        "--cover_threshold",
        "--cover_threshold_default",
        "--iou_threshold",
        dest="cover_threshold_default",
        type=float,
        default=0.30,
        help="Default remove threshold for all classes (fraction 0..1).",
    )

    ap.add_argument(
        "--trama_class",
        type=str,
        default="trama",
        help="Class token (usually numeric id) that corresponds to 'trama'. Example: '2'",
    )
    ap.add_argument(
        "--cover_threshold_trama",
        type=float,
        default=0.60,
        help="Remove threshold for trama class (fraction 0..1).",
    )

    ap.add_argument("--segmentation", type=str, default="quickshift", choices=["quickshift", "slic"])
    ap.add_argument("--replacement_prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--superpixel_scale", type=float, default=1.0)

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save_debug_superpixels", action="store_true")

    # NEW FLAG
    ap.add_argument(
        "--original",
        action="store_true",
        help="If set, also copy the original images and labels into the output dataset (same split).",
    )

    # splitting knobs
    ap.add_argument("--min_component_pixels", type=int, default=200)
    ap.add_argument("--min_component_w_px", type=int, default=4)
    ap.add_argument("--min_component_h_px", type=int, default=4)
    ap.add_argument("--component_pad_px", type=int, default=2)
    ap.add_argument("--connectivity", type=int, default=2, choices=[1, 2])

    args = ap.parse_args()

    cfg = Config(
        dataset_folder=Path(args.dataset_folder),
        backgrounds_folder=Path(args.backgrounds_folder),
        output_folder=Path(args.output_folder),
        num_permutations=args.num_permutations,
        cover_threshold_default=float(args.cover_threshold_default),
        trama_class=str(args.trama_class),
        cover_threshold_trama=float(args.cover_threshold_trama),
        segmentation=args.segmentation,
        replacement_prob=args.replacement_prob,
        superpixel_scale=args.superpixel_scale,
        seed=args.seed,
        verbose=args.verbose,
        save_debug_superpixels=args.save_debug_superpixels,
        save_debug_perm_bboxes=True,
        keep_originals=bool(args.original),
        min_component_pixels=int(args.min_component_pixels),
        min_component_w_px=int(args.min_component_w_px),
        min_component_h_px=int(args.min_component_h_px),
        component_pad_px=int(args.component_pad_px),
        connectivity=int(args.connectivity),
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

    if not (0.0 <= cfg.cover_threshold_default <= 1.0):
        print("ERROR: cover_threshold_default must be between 0 and 1.", flush=True)
        sys.exit(1)
    if not (0.0 <= cfg.cover_threshold_trama <= 1.0):
        print("ERROR: cover_threshold_trama must be between 0 and 1.", flush=True)
        sys.exit(1)

    print("=== Config ===", flush=True)
    print(f"dataset_folder          : {cfg.dataset_folder}", flush=True)
    print(f"backgrounds_folder      : {cfg.backgrounds_folder} ({len(bg_paths)} images)", flush=True)
    print(f"output_folder           : {cfg.output_folder}", flush=True)
    print(f"num_permutations/bg     : {cfg.num_permutations}", flush=True)
    print(
        f"TOTAL perms per image   : {len(bg_paths)} * {cfg.num_permutations} = {len(bg_paths) * cfg.num_permutations}",
        flush=True,
    )
    print(
        f"thresholds              : default={cfg.cover_threshold_default:.2f} | "
        f"trama(class={cfg.trama_class})={cfg.cover_threshold_trama:.2f}",
        flush=True,
    )
    print(f"segmentation            : {cfg.segmentation}", flush=True)
    print(f"replacement_prob        : {cfg.replacement_prob}", flush=True)
    print(f"superpixel_scale        : {cfg.superpixel_scale}", flush=True)
    print(f"seed                    : {cfg.seed}", flush=True)
    print(f"verbose                 : {cfg.verbose}", flush=True)
    print(f"save_debug_superpixels  : {cfg.save_debug_superpixels}", flush=True)
    print(f"keep_originals (--original): {cfg.keep_originals}", flush=True)
    print(
        f"split: min_pixels={cfg.min_component_pixels}, "
        f"min_w={cfg.min_component_w_px}, min_h={cfg.min_component_h_px}, "
        f"pad={cfg.component_pad_px}, conn={cfg.connectivity}",
        flush=True,
    )
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
