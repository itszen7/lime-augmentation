"""Image loading, LIME superpixel segmentation, random replacement masks and debug images."""

import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import find_boundaries


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_and_resize_background(bg_path: Path, size_wh: Tuple[int, int]) -> np.ndarray:
    bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize(size_wh, Image.BILINEAR)
    return np.array(bg)


def get_segmentation_fn(segmentation: str, superpixel_scale: float) -> SegmentationAlgorithm:
    segmentation = segmentation.lower().strip()
    scale = max(0.1, float(superpixel_scale))

    if segmentation == "quickshift":
        # bigger superpixels => larger kernel_size & max_dist
        kernel_size = int(round(16 * scale))
        max_dist = int(round(1400 * scale))
        ratio = 0.3

        print(
            f"[Segmentation] quickshift | superpixel_scale={scale} -> "
            f"kernel_size={kernel_size}, max_dist={max_dist}, ratio={ratio}",
            flush=True,
        )
        return SegmentationAlgorithm("quickshift", kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)

    if segmentation == "slic":
        base_segments = 120
        n_segments = max(10, int(round(base_segments / (scale**2))))
        compactness = 10
        sigma = 1

        print(
            f"[Segmentation] slic | superpixel_scale={scale} -> "
            f"n_segments={n_segments}, compactness={compactness}, sigma={sigma}",
            flush=True,
        )
        return SegmentationAlgorithm("slic", n_segments=n_segments, compactness=compactness, sigma=sigma)

    raise ValueError("segmentation must be 'quickshift' or 'slic'")


def make_random_replacement_mask(segments: np.ndarray, replacement_prob: float, rng: random.Random) -> np.ndarray:
    """Pick a random subset of superpixels to replace."""
    superpixel_ids = np.unique(segments)
    n = len(superpixel_ids)
    if n <= 0:
        # should never happen, but be safe
        return np.zeros_like(segments, dtype=bool)

    replace_flags = np.array([rng.random() < replacement_prob for _ in range(n)], dtype=bool)
    if not replace_flags.any():
        replace_flags[rng.randrange(n)] = True

    replaced_ids = superpixel_ids[replace_flags]
    return np.isin(segments, replaced_ids)


def apply_background_on_mask(image_rgb: np.ndarray, background_rgb: np.ndarray, replaced_mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    out[replaced_mask] = background_rgb[replaced_mask]
    return out


def load_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def save_superpixel_debug_image(img_rgb: np.ndarray, segments: np.ndarray, out_path: Path, font_size: int = 18) -> None:
    """
    grayscale base + black boundaries + yellow id at centroid
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = Image.fromarray(img_rgb).convert("L").convert("RGB")
    arr = np.array(base)

    boundaries = find_boundaries(segments, mode="outer")
    arr[boundaries] = (0, 0, 0)

    dbg = Image.fromarray(arr)
    draw = ImageDraw.Draw(dbg)
    font = load_font(font_size)

    uniq = np.unique(segments)
    id_map = {int(old): int(i) for i, old in enumerate(uniq)}

    for old_id in uniq:
        ys, xs = np.where(segments == old_id)
        if xs.size == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        text = str(id_map[int(old_id)])

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((cx + dx, cy + dy), text, fill=(0, 0, 0), font=font)
        draw.text((cx, cy), text, fill=(255, 255, 0), font=font)

    dbg.save(out_path)
