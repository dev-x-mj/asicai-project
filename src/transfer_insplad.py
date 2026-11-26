"""
Prueba de TRANSFERENCIA: ¿un detector de elementos de línea entrenado con datos públicos
(InsPLAD, 17 clases, fotos a color de otro país) reconoce algo en TUS fotos de postes?

  python src/transfer_insplad.py --out data/real_private --scene 9 --model models/detector-1280-int8.onnx
  python src/transfer_insplad.py --out data/real_private --scene 9 --model models/detector-640-int8.onnx
  python src/transfer_insplad.py --out data/real_private --summarize data/real_private/transfer/detector-1280-int8_conf25/plantilla_conteo.csv

Usa la cámara VISUAL (_V): el modelo se entrenó con imágenes a color, no térmicas.
Todo es LOCAL. Las salidas (con fotos de clientes) NO se suben a ningún repositorio.
Los pesos del modelo son de uso NO COMERCIAL (derivados de InsPLAD): no van al repo.
"""
import argparse
import ast
import csv
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------- pre / postproceso
def letterbox(img_bgr, size):
    """Redimensiona sin deformar y rellena a size x size. Devuelve (tensor, escala, pad_x, pad_y)."""
    h, w = img_bgr.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, px, py


def decode(out, size, r, px, py, W, H, conf=0.25, nms_iou=0.45, nc=17):
    """
    Salida del modelo: (4+nc, N) = cx,cy,w,h y luego un puntaje por clase (sin 'objectness').
    Devuelve lista de dicts con bbox (x1,y1,x2,y2) en píxeles de la imagen ORIGINAL.
    """
    out = np.asarray(out, np.float32)
    if out.shape[0] != 4 + nc and out.shape[1] == 4 + nc:   # por si viene transpuesta (N, 4+nc)
        out = out.T
    boxes, scores = out[:4].T.copy(), out[4:].T
    if boxes.max() <= 2.0:                   # por si las coordenadas vienen normalizadas
        boxes *= size
    cls = scores.argmax(1)
    sc = scores.max(1)
    keep = sc >= conf
    boxes, cls, sc = boxes[keep], cls[keep], sc[keep]
    dets = []
    for c in np.unique(cls):                 # NMS por clase
        idx = np.where(cls == c)[0]
        xywh = [[float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])] for b in boxes[idx]]
        sel = cv2.dnn.NMSBoxes(xywh, [float(s) for s in sc[idx]], conf, nms_iou)
        for k in np.array(sel).flatten():
            x, y, w, h = xywh[int(k)]
            x1, y1 = (x - px) / r, (y - py) / r
            x2, y2 = (x + w - px) / r, (y + h - py) / r
            dets.append({"cls": int(c), "conf": float(sc[idx][int(k)]),
                         "bbox": (max(0, x1), max(0, y1), min(W, x2), min(H, y2))})
    return sorted(dets, key=lambda d: -d["conf"])


# ----------------------------------------------------------------------- flujo
def class_names(sess, nc):
    meta = sess.get_modelmeta().custom_metadata_map.get("names")
    try:
        d = ast.literal_eval(meta)
        return [d[i] for i in range(nc)]
    except Exception:
        return [f"clase_{i}" for i in range(nc)]


def pick_images(out, scene, folder):
    """Devuelve [(stem, ruta_visual)] de la escena pedida (cámara visual) y las que se omiten."""
    if folder:
        ps = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        return [(p.stem, str(p)) for p in ps], []
    from real_data import load_inventory
    rows = [r for r in load_inventory(Path(out) / "index.csv") if int(r["group"]) == scene]
    ok = [(r["stem"], r["v_path"]) for r in rows if r["v_path"] and Path(r["v_path"]).exists()]
    skipped = [r["stem"] for r in rows if not (r["v_path"] and Path(r["v_path"]).exists())]
    return ok, skipped


def run(a):
    import onnxruntime as ort
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else a.size
    nc = int(sess.get_outputs()[0].shape[1]) - 4 if isinstance(sess.get_outputs()[0].shape[1], int) else 17
    names = class_names(sess, nc)
    tag = Path(a.model).stem
    dest = Path(a.out) / "transfer" / f"{tag}_conf{int(round(a.conf * 100)):02d}"   # una carpeta por modelo y umbral: no se pisan resultados
    (dest / "overlays").mkdir(parents=True, exist_ok=True)

    imgs, skipped = pick_images(a.out, a.scene, a.folder)
    if not imgs:
        raise SystemExit("No hay fotos visuales para esa escena.")
    rows, per_class = [], {}
    for stem, path in imgs:
        img = cv2.imread(path)
        H, W = img.shape[:2]
        x, r, px, py = letterbox(img, size)
        dets = decode(sess.run(None, {inp.name: x})[0][0], size, r, px, py, W, H, a.conf, nc=nc)
        vis = img.copy()
        th = max(2, W // 600)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            col = tuple(int(v) for v in np.random.RandomState(d["cls"]).randint(60, 255, 3))
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, th)
            cv2.putText(vis, f"{names[d['cls']]} {d['conf']:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, W / 2500), col, max(1, th - 1), cv2.LINE_AA)
            rows.append({"stem": stem, "clase": names[d["cls"]], "conf": round(d["conf"], 3),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            per_class[names[d["cls"]]] = per_class.get(names[d["cls"]], 0) + 1
        s = 1600 / max(W, H)
        cv2.imwrite(str(dest / "overlays" / f"{stem}.jpg"), cv2.resize(vis, None, fx=s, fy=s) if s < 1 else vis)

    with open(dest / "detecciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "clase", "conf", "x1", "y1", "x2", "y2"])
        w.writeheader(); w.writerows(rows)
    # plantilla para que TÚ cuentes a ojo (las fotos no tienen etiquetas)
    with open(dest / "plantilla_conteo.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stem", "detecciones_totales", "correctas", "erroneas", "elementos_visibles_no_detectados", "notas"])
        for stem, _ in imgs:
            w.writerow([stem, sum(1 for r_ in rows if r_["stem"] == stem), "", "", "", ""])

    print(f"Modelo: {tag} | entrada {size}px | confianza >= {a.conf} | fotos: {len(imgs)} (omitidas sin visual: {len(skipped)})")
    print(f"Detecciones totales: {len(rows)} | fotos con al menos 1 detección: {len({r_['stem'] for r_ in rows})} de {len(imgs)}")
    for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print("Revisa:", (dest / "overlays").resolve())
    print("Cuenta a ojo y llena:", (dest / "plantilla_conteo.csv").resolve())


def summarize(path):
    tot = {"correctas": 0, "erroneas": 0, "no_detectados": 0}
    n = 0
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                c, e, m = int(r["correctas"]), int(r["erroneas"]), int(r["elementos_visibles_no_detectados"])
            except ValueError:
                continue
            tot["correctas"] += c; tot["erroneas"] += e; tot["no_detectados"] += m; n += 1
    P = tot["correctas"] / (tot["correctas"] + tot["erroneas"]) if tot["correctas"] + tot["erroneas"] else 0
    R = tot["correctas"] / (tot["correctas"] + tot["no_detectados"]) if tot["correctas"] + tot["no_detectados"] else 0
    print(f"Fotos contadas: {n} | correctas {tot['correctas']} | erróneas {tot['erroneas']} | no detectados {tot['no_detectados']}")
    print(f"Precisión aprox.: {P:.2f} | Recall aprox.: {R:.2f}  (conteo manual sobre UNA escena: solo orientativo)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--model", default=None, help="ruta al .onnx del detector")
    ap.add_argument("--scene", type=int, default=9, help="número de escena del catálogo (postes = 9)")
    ap.add_argument("--folder", default=None, help="alternativa: carpeta de imágenes cualquiera")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--summarize", default=None, help="CSV de conteo ya llenado")
    a = ap.parse_args()
    summarize(a.summarize) if a.summarize else run(a)
