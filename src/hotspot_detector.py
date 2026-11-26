"""
hotspot_detector.py
--------------------
Detector de puntos calientes (baseline) sobre imágenes térmicas.

Enfoque: visión por computador clásica (umbralización adaptativa +
componentes conectados), NO deep learning. Es intencional: es el
baseline correcto antes de entrenar/ajustar un modelo YOLO —
permite validar el resto del pipeline (ingestión, severidad, reporte)
sin depender de GPU ni de un dataset etiquetado grande.

El módulo de fine-tuning con YOLO (Ultralytics) ya está documentado y
planificado en docs/NEXT_STEPS.md: se integra en este mismo punto del
pipeline (reemplazando o complementando `detect_hotspots`) en cuanto
se cuente con entorno GPU y/o imágenes reales etiquetadas de ASICAI.
"""
import cv2
import numpy as np


def detect_hotspots(gray_image, percentile=97.5, min_area_px=20, std_k=2.2):
    """Detecta regiones anómalamente calientes en una imagen térmica (escala de grises).

    Estrategia:
    1. Umbral adaptativo por percentil (no un valor fijo) — así el detector
       se adapta a la temperatura ambiente de cada escena, en vez de asumir
       un umbral absoluto que no generaliza entre inspecciones distintas.
    2. Componentes conectados sobre la máscara binaria -> cada blob = 1 candidato.
    3. Filtro de área mínima para descartar ruido.
    4. Clasificación de severidad por intensidad relativa del blob.

    Retorna: lista de detecciones:
      [{"bbox":[x,y,w,h], "cx","cy","area_px","mean_intensity","severity"}]
    """
    # Umbral adaptativo combinando percentil y media+desvío, y exigiendo que el
    # punto realmente sobresalga del fondo (evita marcar componentes a
    # temperatura "normal" como si fueran anomalías).
    percentile_value = np.percentile(gray_image, percentile)
    mean_std_value = gray_image.mean() + std_k * gray_image.std()
    threshold_value = max(percentile_value, mean_std_value)

    mask = (gray_image >= threshold_value).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    detections = []
    for label_id in range(1, num_labels):  # 0 = fondo
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label_id]

        region_mean = float(gray_image[labels == label_id].mean())

        # Severidad relativa a la escena completa (proxy de ΔT)
        scene_max = float(gray_image.max())
        ratio = region_mean / scene_max if scene_max > 0 else 0
        if ratio > 0.92:
            severity = "alta"
        elif ratio > 0.8:
            severity = "media"
        else:
            severity = "baja"

        detections.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "cx": float(cx), "cy": float(cy),
            "area_px": int(area),
            "mean_intensity": round(region_mean, 1),
            "severity": severity,
        })

    return detections


def draw_detections(bgr_image, detections, color=(0, 0, 255)):
    out = bgr_image.copy()
    severity_colors = {"alta": (0, 0, 255), "media": (0, 165, 255), "baja": (0, 255, 255)}
    for det in detections:
        x, y, w, h = det["bbox"]
        c = severity_colors.get(det["severity"], color)
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        label = f"{det['severity']} ({det['mean_intensity']:.0f})"
        cv2.putText(out, label, (x, max(y - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return out
