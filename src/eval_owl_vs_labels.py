"""
Evalúa las detecciones de OWLv2 (detecciones.csv ya generado) contra TUS etiquetas multiclase,
por clase y por umbral de confianza. No vuelve a cargar el modelo.

  python src/eval_owl_vs_labels.py --out data/real_private
  python src/eval_owl_vs_labels.py --out data/real_private --det data/real_private/transfer/owlv2_conf10/detecciones.csv --iou 0.3

Requiere: etiquetas en <out>/labels_postes/ (review_multiclass.py) y detecciones.csv de zero_shot_owl.py
generado con un umbral bajo (0.10): desde ahí se filtra hacia arriba.
Es una evaluación sobre POCAS fotos de UNA escena: orientativa, no definitiva.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

from labels_multi import IGNORE_CLASS, iou as box_iou, in_ignore, match_detail, read_classes, read_index, read_labels, xyxy_to_norm

# consulta en inglés (la que usó OWLv2)  ->  nombre de clase en tu lista
QUERY_TO_CLASS = {"utility pole": "poste", "crossarm": "cruceta", "insulator": "aislador",
                  "pole-mounted transformer": "transformador",
                  # descripciones alternativas (varias por clase: se fusionan)
                  "metal crossarm": "cruceta", "steel cross arm bracket": "cruceta", "galvanized steel beam": "cruceta",
                  "concrete utility pole": "poste", "porcelain insulator": "aislador"}


def merge_nms(items, thr=0.5):
    """items = [(conf, caja_norm)]; deja una sola caja cuando varias descripciones marcan lo mismo."""
    kept = []
    for cf, b in sorted(items, key=lambda x: -x[0]):
        if all(box_iou(b, k) < thr for _, k in kept):
            kept.append((cf, b))
    return [b for _, b in kept]


def load_dets(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d.setdefault(r["stem"], []).append((r["clase"], float(r["conf"]),
                                                (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))))
    return d


def evaluate(out, det_csv, thresholds=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5), iou=0.3):
    out = Path(out)
    classes = read_classes(out / "classes_postes.txt")
    lab_dir = out / "labels_postes"
    reviewed = (lab_dir / "reviewed.txt").read_text().split()
    vpath = {r["stem"]: r["v_path"] for r in read_index(out)}
    dets = load_dets(det_csv)
    rows = []
    for ci, cname in enumerate(classes):
        queries = [q for q, c in QUERY_TO_CLASS.items() if c == cname]
        if not queries:
            continue
        for thr in thresholds:
            TP = FP = FN = n_gt = n_ign = 0
            for stem in reviewed:
                with Image.open(vpath[stem]) as im:
                    W, H = im.size
                labs = read_labels(lab_dir / f"{stem}.txt")
                gts = [b[1:] for b in labs if b[0] == ci]
                ign = [b[1:] for b in labs if b[0] < len(classes) and classes[b[0]] == IGNORE_CLASS]
                prs = merge_nms([(cf, xyxy_to_norm(b, W, H)) for q, cf, b in dets.get(stem, []) if q in queries and cf >= thr])
                ok, fn = match_detail(prs, gts, iou)
                extra = [p_ for k, p_ in enumerate(prs) if k not in ok]
                neutral = sum(1 for p_ in extra if in_ignore(p_, ign))      # cae sobre un objeto "dudoso": no cuenta
                TP += len(ok); FP += len(extra) - neutral; FN += fn; n_gt += len(gts); n_ign += neutral
            P = TP / (TP + FP) if TP + FP else 0.0
            R = TP / (TP + FN) if TP + FN else 0.0
            F1 = 2 * P * R / (P + R) if P + R else 0.0
            rows.append({"clase": cname, "umbral": thr, "fotos": len(reviewed), "cajas_reales": n_gt,
                         "TP": TP, "FP": FP, "FN": FN, "ignoradas_dudoso": n_ign, "precision": round(P, 3), "recall": round(R, 3), "f1": round(F1, 3)})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--det", default=None)
    ap.add_argument("--iou", type=float, default=0.3)
    a = ap.parse_args()
    det = a.det or str(Path(a.out) / "transfer" / "owlv2_conf10" / "detecciones.csv")
    rows = evaluate(a.out, det, iou=a.iou)
    (Path(a.out) / "tablas").mkdir(exist_ok=True)
    with open(Path(a.out) / "tablas" / "owl_vs_etiquetas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"IoU >= {a.iou} | fotos etiquetadas: {rows[0]['fotos']}")
    print("(las cajas del modelo que caen sobre objetos marcados 'dudoso' no se cuentan como error)")
    print(f"{'clase':<14}{'umbral':>7}{'reales':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>7}{'recall':>8}{'F1':>6}{'ignor.':>8}")
    for r in rows:
        print(f"{r['clase']:<14}{r['umbral']:>7}{r['cajas_reales']:>8}{r['TP']:>5}{r['FP']:>5}{r['FN']:>5}"
              f"{r['precision']:>7}{r['recall']:>8}{r['f1']:>6}{r['ignoradas_dudoso']:>8}")
    best = {}
    for r in rows:
        if r["cajas_reales"] and (r["clase"] not in best or r["f1"] > best[r["clase"]]["f1"]):
            best[r["clase"]] = r
    print("\nMejor umbral por clase (por F1):", {k: (v["umbral"], v["f1"]) for k, v in best.items()})
    print("Guardado en:", (Path(a.out) / "tablas" / "owl_vs_etiquetas.csv").resolve())