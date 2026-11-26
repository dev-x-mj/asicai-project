"""Utilidades autónomas para etiquetas MULTICLASE (formato YOLO: clase cx cy w h, normalizado 0-1)."""
import csv
from pathlib import Path

DEFAULT_CLASSES = ["poste", "cruceta", "aislador", "transformador", "dudoso"]
IGNORE_CLASS = "dudoso"   # zona neutra: objetos que no se sabe nombrar; no cuentan como acierto ni como falso positivo


def read_classes(path):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(DEFAULT_CLASSES) + "\n", encoding="utf-8")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_index(out):
    with open(Path(out) / "index.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_scene_images(out, scene, limit=None):
    """[(stem, ruta_visual)] de la escena, solo con foto visual; si hay limit, repartidas parejo."""
    rows = [r for r in read_index(out) if int(r["group"]) == scene and r["v_path"] and Path(r["v_path"]).exists()]
    rows.sort(key=lambda r: r["datetime"])
    if limit and len(rows) > limit:
        idx = sorted({round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)}) if limit > 1 else [len(rows) // 2]
        rows = [rows[i] for i in idx]
    return [(r["stem"], r["v_path"]) for r in rows]


def read_labels(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        s = line.split()
        if len(s) >= 5:
            out.append((int(s[0]), *[float(v) for v in s[1:5]]))
    return out


def write_labels(path, boxes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c, cx, cy, w, h in boxes:
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def xyxy_to_norm(b, W, H):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def norm_to_xyxy(b, W, H):
    cx, cy, w, h = b
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def iou(a, b):
    """a, b = (cx, cy, w, h) normalizados."""
    ax1, ay1, ax2, ay2 = norm_to_xyxy(a, 1, 1)
    bx1, by1, bx2, by2 = norm_to_xyxy(b, 1, 1)
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def match_detail(preds, gts, thr):
    """Emparejamiento voraz por IoU. Devuelve (índices de preds acertadas, nº de reales sin acertar)."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    up, ug = set(), set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j)
    return up, len(gts) - len(ug)


def match(preds, gts, thr):
    """Devuelve (tp, fp, fn)."""
    up, fn = match_detail(preds, gts, thr)
    return len(up), len(preds) - len(up), fn


def in_ignore(pred, ign, thr=0.3):
    """¿La caja predicha cae sobre una zona 'dudoso'? (IoU >= thr o su centro está dentro de la zona)."""
    cx, cy = pred[0], pred[1]
    for g in ign:
        x1, y1, x2, y2 = norm_to_xyxy(g, 1, 1)
        if iou(pred, g) >= thr or (x1 <= cx <= x2 and y1 <= cy <= y2):
            return True
    return False
