"""Dataset traversal helpers for YOLO-style folders (images/<split>, labels/<split>)."""

from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_split_dirs(dataset_folder: Path) -> List[Path]:
    images_root = dataset_folder / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing folder: {images_root}")

    split_dirs = []
    for split in ["train", "val", "test"]:
        p = images_root / split
        if p.exists() and p.is_dir():
            split_dirs.append(p)

    if not split_dirs:
        raise FileNotFoundError(f"No split dirs found under: {images_root} (expected train/val/test)")
    return split_dirs


def list_images_in_dir(split_dir: Path) -> List[Path]:
    imgs = [p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    return imgs


def get_label_path_for_image(img_path: Path) -> Path:
    # .../dataset/images/train/img123.jpg  ->  .../dataset/labels/train/img123.txt
    return img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt")
