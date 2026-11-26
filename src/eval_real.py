"""
Compara predicciones contra etiquetas revisadas (ground truth real).

  # Baseline (sus predicciones son labels_pre) sobre las fotos de validación:
  python src/eval_real.py --out data/real_private --pred-dir data/real_private/labels_pre --stems data/real_private/val_stems.txt
  # YOLO (después de train_yolo.py --predict):
  python src/eval_real.py --out data/real_private --pred-dir data/real_private/pred_yolo --stems data/real_private/val_stems.txt

Un acierto (TP) = caja predicha con IoU >= umbral con una caja real (cada caja real se usa una vez).
Con objetos de pocos píxeles el IoU es muy exigente; por eso se reportan varios umbrales.
"""
import argparse
from pathlib import Path

from yolo_io import iou_xywhn, read_labels


def match(preds, gts, thr):
    """Emparejamiento voraz por IoU descendente. Devuelve (tp, fp, fn)."""
    pairs = sorted(((iou_xywhn(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    up, ug, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def evaluate(gt_dir, pred_dir, stems, thr, conf_min=0.0):
    TP = FP = FN = 0
    for s in stems:
        gts = read_labels(Path(gt_dir) / f"{s}.txt")
        prs = [b[:4] for b in read_labels(Path(pred_dir) / f"{s}.txt", with_conf=True) if b[4] >= conf_min]
        tp, fp, fn = match(prs, gts, thr)
        TP += tp; FP += fp; FN += fn
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": round(P, 3), "recall": round(R, 3), "f1": round(F1, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--stems", default=None, help="archivo con stems a evaluar (por defecto: todas las revisadas)")
    ap.add_argument("--conf-min", type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out)
    stems = Path(a.stems).read_text().split() if a.stems else (out / "reviewed.txt").read_text().split()
    n_gt = sum(len(read_labels(out / "labels_reviewed" / f"{s}.txt")) for s in stems)
    print(f"Fotos evaluadas: {len(stems)} | cajas reales (GT): {n_gt}")
    for thr in (0.1, 0.3, 0.5):
        print(f"IoU>={thr}: {evaluate(out / 'labels_reviewed', a.pred_dir, stems, thr, a.conf_min)}")
