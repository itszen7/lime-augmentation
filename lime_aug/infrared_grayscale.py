#!/usr/bin/env python3
"""
Convert images to an "infrared-like" grayscale look (high-contrast B/W),
without losing resolution. Optionally save lossless as PNG.

Examples:
  python -m lime_aug.infrared_grayscale --input ./images --output ./images_ir --recursive --style enhanced --out-format png
  python -m lime_aug.infrared_grayscale --input ./images --output ./images_ir --style basic --out-format keep
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageOps, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def infrared_gray(img: Image.Image, style: str, denoise: float) -> Image.Image:
    """
    style:
      - basic: plain grayscale
      - enhanced: grayscale + autocontrast + equalize (often closer to IR-ish B/W)
    denoise:
      - 0 disables
      - >0 applies a very light median filter (helps with sensor noise)
    """
    # Convert to grayscale (luminance)
    g = ImageOps.grayscale(img)

    if denoise and denoise > 0:
        # Median filter size must be odd integer; map denoise ~ [0..1] -> [3..7]
        k = 3 if denoise < 0.34 else 5 if denoise < 0.67 else 7
        g = g.filter(ImageFilter.MedianFilter(size=k))

    if style == "basic":
        return g

    if style == "enhanced":
        # Stretch tones to full range; then equalize for punchy IR-ish contrast
        g = ImageOps.autocontrast(g, cutoff=0)
        g = ImageOps.equalize(g)
        # Slight sharpening can help the “crisp” IR look
        g = g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        return g

    raise ValueError(f"Unknown style: {style}")


def save_image(img: Image.Image, out_path: Path, out_format: str, keep_exif: Optional[bytes]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    fmt = None

    if out_format == "png":
        fmt = "PNG"
        out_path = out_path.with_suffix(".png")
        img.save(out_path, format=fmt, optimize=True)
        return

    if out_format == "jpg":
        fmt = "JPEG"
        out_path = out_path.with_suffix(".jpg")
        # High-quality JPEG settings (still lossy, but very good)
        img.save(
            out_path,
            format=fmt,
            quality=95,
            subsampling=0,   # keep full chroma sampling (less artifacting)
            optimize=True,
            exif=keep_exif if keep_exif else None,
        )
        return

    if out_format == "keep":
        # Save in the same type as the input file based on extension.
        if suffix in {".jpg", ".jpeg"}:
            img.save(
                out_path,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=True,
                exif=keep_exif if keep_exif else None,
            )
        elif suffix == ".png":
            img.save(out_path, format="PNG", optimize=True)
        elif suffix in {".tif", ".tiff"}:
            img.save(out_path, format="TIFF")
        elif suffix == ".webp":
            img.save(out_path, format="WEBP", quality=95, method=6)
        else:
            # fallback to PNG lossless
            img.save(out_path.with_suffix(".png"), format="PNG", optimize=True)
        return

    raise ValueError("--out-format must be one of: keep, png, jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=str, required=True, help="Input folder of images")
    ap.add_argument("--output", "-o", type=str, required=True, help="Output folder")
    ap.add_argument("--recursive", action="store_true", help="Process subfolders too")
    ap.add_argument("--style", choices=["basic", "enhanced"], default="enhanced", help="Conversion style")
    ap.add_argument("--denoise", type=float, default=0.0, help="0..1 light denoise (median filter). Default 0")
    ap.add_argument(
        "--out-format",
        choices=["keep", "png", "jpg"],
        default="png",
        help="Output format. png is lossless; keep preserves original extension; jpg is high-quality JPEG.",
    )
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)

    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {in_root}")

    count = 0
    for img_path in iter_images(in_root, args.recursive):
        rel = img_path.relative_to(in_root)
        out_path = out_root / rel

        try:
            with Image.open(img_path) as im:
                # Preserve EXIF when possible (mostly relevant for JPEG)
                exif_bytes = im.info.get("exif", None)

                # Ensure consistent mode (some formats come as palette/CMYK/etc.)
                im = im.convert("RGB") if im.mode not in ("RGB", "L") else im

                converted = infrared_gray(im, style=args.style, denoise=args.denoise)
                save_image(converted, out_path, out_format=args.out_format, keep_exif=exif_bytes)

            count += 1
        except Exception as e:
            print(f"[WARN] Failed on {img_path}: {e}")

    print(f"Done. Converted {count} images into: {out_root}")


if __name__ == "__main__":
    main()
