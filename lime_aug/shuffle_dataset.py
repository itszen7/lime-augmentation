#!/usr/bin/env python3
"""
Shuffle a YOLOv8 dataset into train/val/test while preserving per-defect ratios
(i.e., approximately 90/5/5 of the images containing each defect).

Supports multi-label images (an image can have multiple defects).

Input structure:
  dataset/
    images/train, images/val, images/test
    labels/train, labels/val, labels/test

Output structure (same):
  out_dataset/
    images/train|val|test
    labels/train|val|test
"""

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import yaml  # optional; only used if --data-yaml is provided
except Exception:
    yaml = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Helpers: IO + dataset scanning
# -----------------------------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def corresponding_label_path(img_path: Path) -> Path:
    # Replace ".../images/<split>/x.ext" -> ".../labels/<split>/x.txt"
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        raise ValueError(f"Image path does not contain an 'images' directory: {img_path}")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def scan_all_images(dataset_root: Path) -> List[Path]:
    img_root = dataset_root / "images"
    if not img_root.exists():
        raise SystemExit(f"Missing folder: {img_root}")

    imgs: List[Path] = []
    for split in ["train", "val", "test"]:
        split_dir = img_root / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if is_image(p):
                imgs.append(p)

    if not imgs:
        raise SystemExit(f"No images found under: {img_root}/(train|val|test)")
    return sorted(imgs)


def read_present_classes(label_path: Path) -> List[int]:
    """Return list of class ids present in YOLO label file."""
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    classes = set()
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
            classes.add(cls)
        except Exception:
            continue
    return sorted(classes)


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    ensure_parent(dst)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


# -----------------------------
# Multi-label iterative split (explicit counts)
# -----------------------------
def iterative_split_indices_count(
    y: List[List[int]],
    test_count: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    n = len(y)
    if n == 0:
        return [], []

    test_count = max(0, min(n, int(test_count)))
    if test_count == 0:
        return list(range(n)), []

    L = len(y[0]) if y else 0

    total_pos = [0] * L
    for i in range(n):
        for l in range(L):
            total_pos[l] += y[i][l]

    frac = test_count / n
    desired_test = [int(round(frac * total_pos[l])) for l in range(L)]
    current_test = [0] * L

    unassigned = set(range(n))
    test_set: List[int] = []

    def deficit_gain(sample_idx: int) -> int:
        gain = 0
        for l in range(L):
            if y[sample_idx][l] == 1 and current_test[l] < desired_test[l]:
                gain += 1
        return gain

    while len(test_set) < test_count and unassigned:
        needed_labels = [l for l in range(L) if current_test[l] < desired_test[l]]

        pick_label = None
        if needed_labels:
            remaining_counts = []
            for l in needed_labels:
                c = 0
                for i in unassigned:
                    c += y[i][l]
                if c > 0:
                    remaining_counts.append((c, l))
            if remaining_counts:
                remaining_counts.sort()
                pick_label = remaining_counts[0][1]

        if pick_label is not None:
            candidates = [i for i in unassigned if y[i][pick_label] == 1]
            if not candidates:
                candidates = list(unassigned)
        else:
            candidates = list(unassigned)

        candidates = list(candidates)
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: (deficit_gain(i), sum(y[i])), reverse=True)

        chosen = candidates[0]
        test_set.append(chosen)
        for l in range(L):
            current_test[l] += y[chosen][l]
        unassigned.remove(chosen)

    train_set = sorted(unassigned)
    test_set = sorted(test_set)
    return train_set, test_set


def split_three_way(
    y: List[List[int]],
    seed: int,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> Tuple[List[int], List[int], List[int]]:
    n = len(y)
    if n == 0:
        return [], [], []

    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Fractions must sum to 1.0")

    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))

    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    if n_val + n_test >= n:
        if n >= 3:
            n_val = 1
            n_test = 1
        else:
            n_val = min(n_val, n)
            n_test = min(n_test, n - n_val)

    temp_count = n_val + n_test
    train_idx, temp_idx = iterative_split_indices_count(y, test_count=temp_count, seed=seed)

    y_temp = [y[i] for i in temp_idx]
    if not y_temp:
        return train_idx, [], []

    val_temp_idx, test_temp_idx = iterative_split_indices_count(y_temp, test_count=n_test, seed=seed + 1)
    val_idx = [temp_idx[i] for i in val_temp_idx]
    test_idx = [temp_idx[i] for i in test_temp_idx]

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


# -----------------------------
# Reporting
# -----------------------------
def per_label_image_counts(indices: Sequence[int], label_sets: List[List[int]], num_labels: int) -> List[int]:
    counts = [0] * num_labels
    for i in indices:
        for cls in label_sets[i]:
            if 0 <= cls < num_labels:
                counts[cls] += 1
    return counts


def load_names_from_yaml(yaml_path: Path) -> Optional[List[str]]:
    if yaml is None:
        return None
    if not yaml_path.exists():
        return None
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))
    names = data.get("names", None)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        return names
    return None


# -----------------------------
# Robust destination mapping
# -----------------------------
def get_old_split_from_path(dataset_root: Path, img_path: Path) -> str:
    """
    Returns old split name by locating dataset_root/images/<split>/...
    """
    rel = img_path.relative_to(dataset_root)
    parts = rel.parts
    # expected: ("images", "<split>", ...)
    if len(parts) < 3 or parts[0] != "images":
        raise ValueError(f"Unexpected image path (expected dataset/images/<split>/...): {img_path}")
    return parts[1]


def image_rel_within_split(dataset_root: Path, img_path: Path) -> Path:
    """
    Returns path relative to dataset_root/images/<split>/...
    Example: images/train/a/b.jpg -> a/b.jpg
    """
    old_split = get_old_split_from_path(dataset_root, img_path)
    return img_path.relative_to(dataset_root / "images" / old_split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="Path to YOLO dataset root")
    ap.add_argument("--output", required=True, type=str, help="Path to output dataset root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--data-yaml", type=str, default=None, help="Optional data.yaml to get defect names")
    ap.add_argument(
        "--include-background",
        action="store_true",
        help="Treat images with no labels as an extra 'background' class for stratification",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    out_root = Path(args.output)

    # Create split folders upfront (so you SEE them even before copying)
    for sp in ["train", "val", "test"]:
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    imgs = scan_all_images(dataset_root)

    # Build label sets
    label_sets: List[List[int]] = []
    max_cls = -1
    for img_path in imgs:
        lp = corresponding_label_path(img_path)
        cls_list = read_present_classes(lp)
        label_sets.append(cls_list)
        if cls_list:
            max_cls = max(max_cls, max(cls_list))

    if max_cls < 0:
        raise SystemExit("No labels found in dataset. (All label files missing/empty)")

    base_num_labels = max_cls + 1
    names = load_names_from_yaml(Path(args.data_yaml)) if args.data_yaml else None
    if names and len(names) < base_num_labels:
        names = names + [f"class_{i}" for i in range(len(names), base_num_labels)]
    if names is None:
        names = [f"class_{i}" for i in range(base_num_labels)]

    # Build multi-hot vectors
    background_idx = base_num_labels if args.include_background else None
    num_labels = base_num_labels + (1 if args.include_background else 0)
    if args.include_background:
        names = names + ["background"]

    y: List[List[int]] = []
    for cls_list in label_sets:
        vec = [0] * num_labels
        if cls_list:
            for c in cls_list:
                if 0 <= c < base_num_labels:
                    vec[c] = 1
        else:
            if args.include_background:
                vec[background_idx] = 1  # type: ignore
        y.append(vec)

    # Split
    train_idx, val_idx, test_idx = split_three_way(y, seed=args.seed, train_frac=0.90, val_frac=0.05, test_frac=0.05)

    print(f"Total images: {len(imgs)}")
    print(f"Split sizes -> train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    def show(split_name: str, idxs: List[int]):
        counts = per_label_image_counts(idxs, label_sets, base_num_labels)
        total = len(idxs)
        print(f"\n{split_name}: {total} images")
        for i, c in enumerate(counts):
            if c > 0:
                print(f"  {i:>2} {names[i]:<20} -> {c} images")

    show("TRAIN", train_idx)
    show("VAL", val_idx)
    show("TEST", test_idx)

    # Write output dataset (ROBUST PATHS)
    split_plan = [("train", train_idx), ("val", val_idx), ("test", test_idx)]

    written_counts = {"train": 0, "val": 0, "test": 0}

    for new_split, idxs in split_plan:
        for i in idxs:
            src_img = imgs[i]
            src_lbl = corresponding_label_path(src_img)

            # Preserve any nested structure under the split folder
            rel_inside = image_rel_within_split(dataset_root, src_img)  # e.g., sub/a.jpg or a.jpg

            dst_img = out_root / "images" / new_split / rel_inside
            dst_lbl = out_root / "labels" / new_split / rel_inside.with_suffix(".txt")

            copy_or_move(src_img, dst_img, move=args.move)

            if src_lbl.exists():
                copy_or_move(src_lbl, dst_lbl, move=args.move)
            else:
                ensure_parent(dst_lbl)
                dst_lbl.write_text("", encoding="utf-8")

            written_counts[new_split] += 1

    print("\nWritten files per split:", written_counts)
    print(f"Done. Output written to: {out_root}")
    if args.move:
        print("Note: files were MOVED from the original dataset.")
    else:
        print("Note: files were COPIED; original dataset unchanged.")


if __name__ == "__main__":
    main()
