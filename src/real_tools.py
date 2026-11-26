"""Funciones de apoyo para resultados_reales.ipynb (el notebook solo las llama)."""
from pathlib import Path

from real_baseline import bbox_to_yolo, detect_simple
from real_data import load_thermal
from eval_real import match
from yolo_io import read_labels


def cargar_arrays(out, stems, sdk_exe=None, temp_dir=None):
    """Lee (o toma de la caché) la matriz térmica de cada foto. {stem: array}."""
    out = Path(out)
    import csv
    with open(out / "index.csv", encoding="utf-8") as f:
        t_path = {r["stem"]: r["t_path"] for r in csv.DictReader(f)}
    return {s: load_thermal(t_path[s], sdk_exe=sdk_exe, cache_dir=out / "temp_cache", temp_dir=temp_dir)[0] for s in stems}


def predecir(arr, percentile=97.5, std_k=2.2, min_area_px=20, max_area_frac=None):
    dets, _ = detect_simple(arr, percentile, std_k, min_area_px, max_area_frac)
    H, W = arr.shape
    return [bbox_to_yolo(d["bbox"], W, H) for d in dets]


def eval_baseline(arrays, gt_dir, stems, iou=0.3, **params):
    """Métricas del baseline con ciertos parámetros sobre 'stems' (contra etiquetas revisadas)."""
    TP = FP = FN = 0
    for s in stems:
        tp, fp, fn = match(predecir(arrays[s], **params), read_labels(Path(gt_dir) / f"{s}.txt"), iou)
        TP += tp; FP += fp; FN += fn
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": round(P, 3), "recall": round(R, 3), "f1": round(F1, 3)}


def errores_por_foto(arrays, gt_dir, stems, iou=0.3, **params):
    """Lista de (stem, tp, fp, fn) ordenada de peor a mejor (fp+fn)."""
    filas = []
    for s in stems:
        tp, fp, fn = match(predecir(arrays[s], **params), read_labels(Path(gt_dir) / f"{s}.txt"), iou)
        filas.append((s, tp, fp, fn))
    return sorted(filas, key=lambda r: -(r[2] + r[3]))
