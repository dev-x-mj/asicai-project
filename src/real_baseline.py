"""
Detector baseline sobre fotos reales + pre-etiquetas en formato YOLO.

Replica la lógica descrita en tu notebook (umbral = max(percentil, media + k*std),
componentes conectados, área mínima). Si prefieres tu `detect_hotspots`, solo cambia
la llamada en run().   [No vi tu hotspot_detector.py, por eso esta versión es autónoma.]

Uso típico (desde la carpeta asicai-poc):
  python src/real_baseline.py --root "D:/fotos_publicas" --out data/real_private
  python src/real_baseline.py --root "D:/fotos_publicas" --out data/real_private --temp-dir data/real_private/temp_c_tiff

Salidas en --out  (CONFIDENCIAL: no subir a GitHub ni a ningún chat):
  index.csv        una fila por foto (grupo, ruta, tamaño...)
  detections.csv   una fila por foto con n.º de detecciones y umbral
  render/          imagen con paleta uniforme (la que se usa para revisar/entrenar)
  overlays/        render + cajas dibujadas (para mirar)
  labels_pre/      pre-etiquetas YOLO del baseline (NO editar: son las "predicciones del baseline")
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from real_data import build_inventory, load_thermal, render_uniform, save_inventory


def detect_simple(arr, percentile=97.5, std_k=2.2, min_area_px=20, max_area_frac=None):
    """Devuelve (lista de detecciones, umbral usado). bbox = (x, y, w, h) en píxeles."""
    arr = arr.astype(np.float32)
    if arr.std() < 1e-6:
        return [], float(arr.mean())
    thr = max(float(np.percentile(arr, percentile)), float(arr.mean() + std_k * arr.std()))
    mask = (arr >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = arr.shape
    dets = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < min_area_px:
            continue
        if max_area_frac and a > max_area_frac * H * W:
            continue
        reg = arr[lab == i]
        dets.append({"bbox": (x, y, w, h), "area": a,
                     "max": float(reg.max()), "mean": float(reg.mean())})
    return dets, thr


def bbox_to_yolo(bbox, W, H):
    x, y, w, h = bbox
    return ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)


def write_yolo(path, boxes_norm):
    with open(path, "w") as f:
        for cx, cy, w, h in boxes_norm:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def run(args):
    out = Path(args.out)
    for d in ("render", "overlays", "labels_pre", "temp_cache"):
        (out / d).mkdir(parents=True, exist_ok=True)

    rows = build_inventory(args.root, gap_s=args.gap_s)
    if not rows:
        raise SystemExit("No encontré archivos DJI_*_T.JPG en esa carpeta.")
    save_inventory(rows, out / "index.csv")

    resumen = []
    for r in rows:
        arr, kind = load_thermal(r["t_path"], sdk_exe=args.sdk, cache_dir=out / "temp_cache",
                                 temp_dir=getattr(args, "temp_dir", None))
        H, W = arr.shape
        dets, thr = detect_simple(arr, args.percentile, args.std_k, args.min_area, args.max_area_frac)

        lo, hi = (args.fixed_range if args.fixed_range else (None, None))
        img = render_uniform(arr, lo, hi)
        cv2.imwrite(str(out / "render" / f"{r['stem']}.jpg"), img)

        vis = img.copy()
        for k, d in enumerate(dets, 1):
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(vis, str(k), (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(str(out / "overlays" / f"{r['stem']}.jpg"), vis)
        write_yolo(out / "labels_pre" / f"{r['stem']}.txt", [bbox_to_yolo(d["bbox"], W, H) for d in dets])

        resumen.append({"stem": r["stem"], "group": r["group"], "tipo_dato": kind,
                        "n_detecciones": len(dets), "umbral": round(thr, 2),
                        "max_valor": round(float(arr.max()), 2),
                        "min_valor": round(float(arr.min()), 2)})

    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)

    n = len(rows)
    sin = sum(1 for x in resumen if x["n_detecciones"] == 0)
    print(f"Fotos térmicas: {n} | grupos (escenas): {len({r['group'] for r in rows})}")
    print(f"Tipo de dato: {resumen[0]['tipo_dato']}  (proxy = NO es temperatura; usa --temp-dir (dji-thermal-cli) o --sdk para °C reales)")
    print(f"Fotos sin detecciones: {sin} | con detecciones: {n - sin}")
    print(f"Resultados en: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="carpeta con las fotos (ya descomprimidas)")
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--temp-dir", default=None,
                    help="carpeta con los *_temp_c.tiff generados por dji-thermal-cli (°C reales)")
    ap.add_argument("--sdk", default=None, help="alternativa: ruta a dji_irp(.exe) del DJI Thermal SDK oficial")
    ap.add_argument("--percentile", type=float, default=97.5)
    ap.add_argument("--std_k", type=float, default=2.2)
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--max_area_frac", type=float, default=None,
                    help="descarta regiones que ocupen más de esta fracción de la imagen (p.ej. 0.3)")
    ap.add_argument("--gap_s", type=int, default=90, help="segundos que separan una escena de otra")
    ap.add_argument("--fixed-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="rango fijo de °C para el render (solo con --sdk), p.ej. 20 120")
    run(ap.parse_args())
