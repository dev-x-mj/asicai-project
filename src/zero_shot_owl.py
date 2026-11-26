"""
Detección "sin entrenar" con texto (OWLv2, Google, licencia Apache 2.0): le dices en INGLÉS qué buscar
("utility pole", "insulator"...) y devuelve cajas. No depende de las 17 clases de InsPLAD.

  pip install torch transformers
  python src/zero_shot_owl.py --out data/real_private --scene 9
  python src/zero_shot_owl.py --out data/real_private --scene 9 --queries "utility pole;crossarm;insulator;transformer"

Es una PRUEBA de qué tan bien reconoce elementos de tus fotos de postes, no un sistema validado.
La primera corrida descarga el modelo (~600 MB a 1.2 GB) a la caché de Hugging Face.
Todo es local; las salidas contienen fotos de clientes: no se suben a ningún repositorio.
Los nombres de clase deben confirmarlos técnicos de ASICAI.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEFAULT_QUERIES = ["utility pole", "crossarm", "insulator", "pole-mounted transformer", "power line cable"]


def decode_owl(logits, pred_boxes, S, conf=0.10):
    """
    Decodifica la salida CRUDA de OWLv2 (equivale a lo que hacía post_process_object_detection, que
    cambia de nombre/lugar entre versiones de transformers).
      logits     (N, n_consultas)   puntajes sin normalizar
      pred_boxes (N, 4)             cx, cy, w, h NORMALIZADOS respecto al lienzo cuadrado SxS
    Devuelve (boxes xyxy en píxeles del lienzo cuadrado, scores, labels).
    """
    logits = np.asarray(logits, np.float32)
    pred_boxes = np.asarray(pred_boxes, np.float32)
    best = logits.max(1)
    labels = logits.argmax(1)
    scores = 1.0 / (1.0 + np.exp(-best))
    keep = scores >= conf
    cx, cy, w, h = (pred_boxes[keep, i] * S for i in range(4))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) if keep.any() else np.zeros((0, 4), np.float32)
    return boxes, scores[keep], labels[keep]


def postprocess(boxes, scores, labels, W, H, nms_iou=0.5):
    """
    boxes (N,4) xyxy en la escala del lienzo CUADRADO (OWLv2 rellena la imagen a cuadrado, abajo/derecha).
    Recorta a la imagen real y quita cajas repetidas por clase. Devuelve [(label, score, (x1,y1,x2,y2))].
    """
    out = []
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, np.float32)
    labels = np.asarray(labels, int)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[idx]]
        ok = [i for i, b in enumerate(xywh) if b[2] > 1 and b[3] > 1]
        if not ok:
            continue
        sel = cv2.dnn.NMSBoxes([xywh[i] for i in ok], [float(scores[idx][i]) for i in ok], 0.0, nms_iou)
        for k in np.array(sel).flatten():
            i = ok[int(k)]
            out.append((int(lab), float(scores[idx][i]), tuple(float(v) for v in boxes[idx][i])))
    return sorted(out, key=lambda d: -d[1])


def run(a):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from transfer_insplad import pick_images

    queries = [q.strip() for q in a.queries.split(";") if q.strip()] if a.queries else DEFAULT_QUERIES
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval()

    imgs, skipped = pick_images(a.out, a.scene, a.folder)
    if not imgs:
        raise SystemExit("No hay fotos visuales para esa escena.")
    name = ("owlv2_" + a.tag + "_conf") if a.tag else "owlv2_conf"
    dest = Path(a.out) / "transfer" / f"{name}{int(round(a.conf * 100)):02d}"
    (dest / "overlays").mkdir(parents=True, exist_ok=True)

    rows, per_class = [], {}
    for stem, path in imgs:
        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        S = max(W, H)
        inputs = processor(text=[queries], images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        b_, s_, l_ = decode_owl(outputs.logits[0].numpy(), outputs.pred_boxes[0].numpy(), S, a.conf)
        dets = postprocess(b_, s_, l_, W, H)
        vis = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        th = max(2, W // 600)
        for lab, sc, (x1, y1, x2, y2) in dets:
            name = queries[lab]
            col = tuple(int(v) for v in np.random.RandomState(lab).randint(60, 255, 3))
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, th)
            cv2.putText(vis, f"{name} {sc:.2f}", (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, W / 2500), col, max(1, th - 1), cv2.LINE_AA)
            rows.append({"stem": stem, "clase": name, "conf": round(sc, 3),
                         "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
            per_class[name] = per_class.get(name, 0) + 1
        s = 1600 / max(W, H)
        cv2.imwrite(str(dest / "overlays" / f"{stem}.jpg"), cv2.resize(vis, None, fx=s, fy=s) if s < 1 else vis)
        print(f"  {stem}: {len(dets)} cajas")

    with open(dest / "detecciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "clase", "conf", "x1", "y1", "x2", "y2"])
        w.writeheader(); w.writerows(rows)
    with open(dest / "plantilla_conteo.csv", "w", newline="", encoding="utf-8-sig") as f:   # mismo formato que transfer_insplad
        w = csv.writer(f)
        w.writerow(["stem", "detecciones_totales", "correctas", "erroneas", "elementos_visibles_no_detectados", "notas"])
        for stem, _ in imgs:
            w.writerow([stem, sum(1 for r in rows if r["stem"] == stem), "", "", "", ""])

    print(f"OWLv2 | consultas: {queries} | confianza >= {a.conf} | fotos: {len(imgs)} (omitidas sin visual: {len(skipped)})")
    print(f"Detecciones totales: {len(rows)} | fotos con al menos 1 detección: {len({r['stem'] for r in rows})} de {len(imgs)}")
    for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print("Revisa:", (dest / "overlays").resolve())
    print("Cuenta a ojo y llena:", (dest / "plantilla_conteo.csv").resolve())
    print("Resumen: python src/transfer_insplad.py --summarize", dest / "plantilla_conteo.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--scene", type=int, default=9)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--queries", default=None, help='consultas en inglés separadas por ";"')
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--tag", default=None, help="nombre para la carpeta de resultados (no pisa corridas anteriores)")
    run(ap.parse_args())