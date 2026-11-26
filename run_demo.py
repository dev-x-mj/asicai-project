"""
run_demo.py
------------
Orquesta el pipeline completo de punta a punta:

  1. Genera (o carga) escenas térmicas
  2. Corre el detector de hotspots (baseline CV clásico)
  3. Evalúa contra ground truth (precision/recall/F1)
  4. Guarda imágenes anotadas + un reporte CSV/JSON con todos los hallazgos

Uso:
    python run_demo.py
"""
import os
import sys
import json
import csv
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from thermal_simulator import save_dataset
from hotspot_detector import detect_hotspots, draw_detections
from evaluate import match_detections_to_gt, compute_metrics

BASE_DIR = os.path.dirname(__file__)
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1) Generando set de escenas térmicas sintéticas (hasta tener imágenes reales de ASICAI)...")
    manifest = save_dataset(SAMPLE_DIR, n_images=8)

    all_matches, all_fn, all_fp = [], [], []
    report_rows = []

    print("2) Corriendo detector de hotspots sobre cada escena...")
    for entry in manifest:
        img_path = os.path.join(SAMPLE_DIR, entry["file"])
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        detections = detect_hotspots(gray, percentile=95, min_area_px=20)
        matches, fn, fp = match_detections_to_gt(detections, entry["ground_truth"])

        all_matches.extend(matches)
        all_fn.extend(fn)
        all_fp.extend(fp)

        # Imagen anotada (con paleta de color tipo térmico para que se vea realista)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        annotated = draw_detections(colored, detections)
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{entry['file']}")
        cv2.imwrite(out_path, annotated)

        for det in detections:
            report_rows.append({
                "imagen": entry["file"],
                "severidad": det["severity"],
                "intensidad_media": det["mean_intensity"],
                "area_px": det["area_px"],
                "bbox": det["bbox"],
            })

    metrics = compute_metrics(all_matches, all_fn, all_fp)

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "detecciones.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["imagen", "severidad", "intensidad_media", "area_px", "bbox"])
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n3) RESULTADOS")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nImágenes anotadas guardadas en: {OUTPUT_DIR}")
    print(f"Reporte CSV: {os.path.join(OUTPUT_DIR, 'detecciones.csv')}")
    return metrics


if __name__ == "__main__":
    main()
