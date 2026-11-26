"""Utilidades pequeñas para etiquetas YOLO (clase cx cy w h, todo normalizado 0-1)."""
from pathlib import Path


def read_labels(path, with_conf=False):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            vals = [float(v) for v in parts[1:5]]
            if with_conf:
                vals.append(float(parts[5]) if len(parts) > 5 else 1.0)
            out.append(tuple(vals))
    return out


def write_labels(path, boxes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")


def xywhn_to_xyxy(b, W, H):
    cx, cy, w, h = b[:4]
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def xyxy_to_xywhn(b, W, H):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    return (((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H)


def iou_xywhn(a, b):
    ax1, ay1, ax2, ay2 = xywhn_to_xyxy(a, 1, 1)
    bx1, by1, bx2, by2 = xywhn_to_xyxy(b, 1, 1)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
