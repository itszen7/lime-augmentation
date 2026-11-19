"""YOLO label helpers: read/write label files and convert boxes to pixel coordinates."""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_yolo_labels(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def write_yolo_labels(label_path: Path, lines: List[str]) -> None:
    """
    Always creates the label file.
    If lines empty => writes an empty .txt (valid YOLO empty-label image).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        label_path.write_text("", encoding="utf-8")
    else:
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def yolo_line_to_bbox_px(line: str, img_w: int, img_h: int) -> Tuple[str, Tuple[float, float, float, float]]:
    """
    YOLO: class x_center y_center w h (normalized)
    Returns: (class_str, (x1,y1,x2,y2)) in pixel coords (clamped)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line (expected >=5 tokens): {line!r}")

    cls = parts[0]
    xc, yc, w, h = map(float, parts[1:5])

    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h

    # clamp
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))

    return cls, (x1, y1, x2, y2)


def bbox_mask_coverage(replaced_mask: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """
    coverage = (replaced pixels inside bbox) / (bbox area)
    """
    x1, y1, x2, y2 = bbox_xyxy
    H, W = replaced_mask.shape

    x1i = int(np.floor(x1))
    y1i = int(np.floor(y1))
    x2i = int(np.ceil(x2))
    y2i = int(np.ceil(y2))

    # clamp ints
    x1i = max(0, min(W, x1i))
    x2i = max(0, min(W, x2i))
    y1i = max(0, min(H, y1i))
    y2i = max(0, min(H, y2i))

    if x2i <= x1i or y2i <= y1i:
        return 0.0

    # replaced pixels inside bbox
    inter = int(replaced_mask[y1i:y2i, x1i:x2i].sum())
    box_area = int((x2i - x1i) * (y2i - y1i))
    if box_area <= 0:
        return 0.0

    return inter / box_area
