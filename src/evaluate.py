"""
evaluate.py
------------
Compara las detecciones del modelo contra el ground truth (conocido porque
las escenas son sintéticas) y calcula precisión / recall / F1.

Esto es clave para el informe: en vez de decir "el detector funciona bien",
podemos decir "detectó X de Y puntos calientes reales, con Z falsos positivos".
"""
import math


def match_detections_to_gt(detections, ground_truth, max_distance_px=25):
    """Empareja cada ground truth con la detección más cercana (si existe)."""
    matched_gt_ids = set()
    matched_det_idx = set()
    matches = []

    for gt in ground_truth:
        best_idx, best_dist = None, float("inf")
        for i, det in enumerate(detections):
            if i in matched_det_idx:
                continue
            dist = math.hypot(det["cx"] - gt["cx"], det["cy"] - gt["cy"])
            if dist < best_dist:
                best_dist, best_idx = dist, i

        if best_idx is not None and best_dist <= max_distance_px:
            matches.append({"gt": gt, "detection": detections[best_idx], "distance_px": round(best_dist, 1)})
            matched_gt_ids.add(gt["id"])
            matched_det_idx.add(best_idx)

    false_negatives = [gt for gt in ground_truth if gt["id"] not in matched_gt_ids]
    false_positives = [det for i, det in enumerate(detections) if i not in matched_det_idx]

    return matches, false_negatives, false_positives


def compute_metrics(all_matches, all_fn, all_fp):
    tp = len(all_matches)
    fn = len(all_fn)
    fp = len(all_fp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "verdaderos_positivos": tp,
        "falsos_negativos": fn,
        "falsos_positivos": fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
    }
