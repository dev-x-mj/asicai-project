"""
Prueba EXPLORATORIA de mejora del detector baseline sobre las fotos ya etiquetadas (labels_reviewed).

  python src/sweep_baseline.py --out data/real_private --temp-dir data/real_private/temp_c_tiff

Añade dos filtros al detector actual, sin tocar sus archivos:
  - min_delta: la zona debe superar en al menos X °C la mediana de su entorno inmediato (anillo alrededor de la caja)
  - top_k: se conservan solo las K zonas con mayor diferencia por foto
Imprime la fila del baseline actual (debe coincidir con eval_real.py) y las mejores combinaciones.
OJO: se ajusta sobre las MISMAS fotos con que se mide: los valores son optimistas. Sirve para decidir hacia dónde
mejorar, no como desempeño final.
"""
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

from labels_multi import match, read_index
from real_data import load_thermal


def read_boxes(path):
    p = Path(path)
    if not p.exists():
        return []
    return [tuple(float(v) for v in l.split()[1:5]) for l in p.read_text().splitlines() if len(l.split()) >= 5]


def detect(arr, percentile=97.5, std_k=2.2, min_area=20):
    """Igual que el baseline: umbral = max(percentil, media + k*std) y componentes conectados."""
    arr = arr.astype(np.float32)
    if arr.std() < 1e-6:
        return []
    thr = max(float(np.percentile(arr, percentile)), float(arr.mean() + std_k * arr.std()))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((arr >= thr).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a >= min_area:
            out.append({"bbox": (x, y, w, h), "vmax": float(arr[lab == i].max())})
    return out


def delta_vs_surroundings(arr, bbox, margin=0.6):
    """Máximo de la zona menos la mediana del anillo que la rodea (el anillo excluye la propia caja)."""
    H, W = arr.shape
    x, y, w, h = bbox
    mx, my = int(w * margin) + 3, int(h * margin) + 3
    x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)
    win = arr[y0:y1, x0:x1].copy()
    mask = np.ones(win.shape, bool)
    mask[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    ring = win[mask]
    return float(arr[y:y + h, x:x + w].max() - np.median(ring)) if ring.size else 0.0


def predict(arr, std_k, min_area, min_delta, top_k):
    H, W = arr.shape
    dets = []
    for d in detect(arr, 97.5, std_k, min_area):
        dl = delta_vs_surroundings(arr, d["bbox"])
        if dl >= min_delta:
            dets.append((dl, d["bbox"]))
    dets.sort(key=lambda t: -t[0])
    if top_k:
        dets = dets[:top_k]
    return [((x + w / 2) / W, (y + h / 2) / H, w / W, h / H) for _, (x, y, w, h) in dets]


def evaluate(arrays, gts, cfg, iou):
    TP = FP = FN = 0
    for s, arr in arrays.items():
        tp, fp, fn = match(predict(arr, *cfg), gts[s], iou)
        TP += tp; FP += fp; FN += fn
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return TP, FP, FN, P, R, F


def per_photo(arrays, gts, cfg, iou):
    """{foto: (tp, fp, fn)} para una configuración."""
    return {s: match(predict(arr, *cfg), gts[s], iou) for s, arr in arrays.items()}


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def loso(arrays, gts, group_of, grid):
    """
    Validación cruzada dejando UNA ESCENA fuera: la mejor configuración se elige solo con las demás escenas
    (criterio: F1 con IoU>=0.3, desempate por F1 con IoU>=0.1) y se mide en la escena excluida.
    Devuelve los totales (tp, fp, fn) con IoU 0.3 y 0.1, y cuántas veces salió elegida cada configuración.
    """
    t3 = {c: per_photo(arrays, gts, c, 0.3) for c in grid}
    t1 = {c: per_photo(arrays, gts, c, 0.1) for c in grid}
    tot3, tot1, elegidas = [0, 0, 0], [0, 0, 0], {}
    for g in sorted({group_of[s] for s in arrays}):
        train = [s for s in arrays if group_of[s] != g]
        test = [s for s in arrays if group_of[s] == g]
        def score(c):
            a3 = [sum(t3[c][s][k] for s in train) for k in range(3)]
            a1 = [sum(t1[c][s][k] for s in train) for k in range(3)]
            return (prf(*a3)[2], prf(*a1)[2], -a3[1])
        best = max(grid, key=score)
        elegidas[best] = elegidas.get(best, 0) + 1
        for k in range(3):
            tot3[k] += sum(t3[best][s][k] for s in test)
            tot1[k] += sum(t1[best][s][k] for s in test)
    return tot3, tot1, elegidas


def main(a):
    out = Path(a.out)
    stems = (out / "reviewed.txt").read_text().split()
    tpath = {r["stem"]: r["t_path"] for r in read_index(out)}
    arrays = {s: load_thermal(tpath[s], cache_dir=out / "temp_cache", temp_dir=a.temp_dir)[0] for s in stems}
    gts = {s: read_boxes(out / "labels_reviewed" / f"{s}.txt") for s in stems}
    n_gt = sum(len(v) for v in gts.values())
    print(f"Fotos: {len(stems)} | cajas reales: {n_gt}")
    base = (2.2, 20, 0.0, None)
    print("\nBASELINE ACTUAL (debe coincidir con eval_real.py):")
    for iou in (0.1, 0.3):
        TP, FP, FN, P, R, F = evaluate(arrays, gts, base, iou)
        print(f"  IoU>={iou}: TP {TP} FP {FP} FN {FN} | precision {P:.3f} recall {R:.3f} F1 {F:.3f}")
    rows = []
    for std_k, min_area, min_delta, top_k in itertools.product((2.2, 3.0), (20, 60, 150), (0, 2, 3, 5, 8, 10), (None, 3, 2, 1)):
        cfg = (std_k, min_area, float(min_delta), top_k)
        r1 = evaluate(arrays, gts, cfg, 0.1); r3 = evaluate(arrays, gts, cfg, 0.3)
        rows.append((r3[5], r1[5], cfg, r3, r1))
    grid = [r[2] for r in rows]
    rows.sort(key=lambda r: (-r[0], -r[1]))
    print("\nMEJORES COMBINACIONES (orden por F1 con IoU>=0.3; exploratorio, ajustado sobre las mismas fotos):")
    print(f"{'std_k':>6}{'area':>6}{'dT min':>8}{'top_k':>7} | {'IoU.3: TP/FP/FN':>16}{'prec':>7}{'rec':>7}{'F1':>7} | {'IoU.1: TP/FP/FN':>16}{'prec':>7}{'rec':>7}{'F1':>7}")
    report_loso_args = (a, arrays, gts, grid, stems)
    for _, _, cfg, r3, r1 in rows[:a.top]:
        print(f"{cfg[0]:>6}{cfg[1]:>6}{cfg[2]:>8}{str(cfg[3]):>7} | {f'{r3[0]}/{r3[1]}/{r3[2]}':>16}{r3[3]:>7.2f}{r3[4]:>7.2f}{r3[5]:>7.2f} | {f'{r1[0]}/{r1[1]}/{r1[2]}':>16}{r1[3]:>7.2f}{r1[4]:>7.2f}{r1[5]:>7.2f}")
    report_loso(*report_loso_args)


def report_loso(a, arrays, gts, grid, stems):
    group_of = {r["stem"]: r["group"] for r in read_index(Path(a.out))}
    tot3, tot1, elegidas = loso(arrays, gts, group_of, grid)
    print("\nVALIDACIÓN CRUZADA POR ESCENA (la configuración se elige SIN ver la escena que se mide):")
    for iou, tot in ((0.3, tot3), (0.1, tot1)):
        P, R, F = prf(*tot)
        print(f"  IoU>={iou}: TP {tot[0]} FP {tot[1]} FN {tot[2]} | precision {P:.3f} recall {R:.3f} F1 {F:.3f}")
    print("  Configuraciones elegidas (std_k, area, dT min, top_k): veces")
    for c, n in sorted(elegidas.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {c}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--top", type=int, default=8)
    main(ap.parse_args())