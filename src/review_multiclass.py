"""
Etiquetado MULTICLASE local (OpenCV) sobre la cámara VISUAL de una escena (por defecto la 9 = postes).

  python src/review_multiclass.py --out data/real_private --scene 9 --limit 8

Clases: archivo <out>/classes_postes.txt (una por línea; se crea con una propuesta si no existe).
La lista de clases es una PROPUESTA: debe validarla ASICAI.

Controles:
  1..9                     elegir la clase actual (se muestra arriba)
  clic izq. + arrastrar    dibujar caja de la clase actual
  clic derecho             borrar la caja bajo el cursor
  u                        deshacer la última caja     c   borrar todas
  d / espacio / flecha der guardar y siguiente         a / flecha izq   guardar y anterior
  q / ESC                  guardar y salir
Etiqueta TODOS los elementos visibles de las clases de la lista en cada foto. Si dudas de un elemento,
no lo marques y apúntalo para el técnico. Cuando termines una foto, se guarda en <out>/labels_postes/.
Al salir se imprime cuánto tardaste por foto (dato útil para estimar el esfuerzo de etiquetado).
"""
import argparse
import time
from pathlib import Path

import cv2

from labels_multi import (norm_to_xyxy, pick_scene_images, read_classes, read_labels, write_labels,
                          xyxy_to_norm)

COLORS = [(60, 180, 255), (80, 220, 100), (255, 120, 60), (200, 80, 220), (60, 220, 220),
          (255, 200, 60), (160, 160, 255), (120, 255, 200), (255, 100, 160)]


def box_at(boxes, x, y):
    """Índice de la caja (x1,y1,x2,y2,cls) más pequeña que contiene el punto, o None."""
    best, area = None, None
    for i, (x1, y1, x2, y2, _) in enumerate(boxes):
        if x1 <= x <= x2 and y1 <= y <= y2:
            a = (x2 - x1) * (y2 - y1)
            if best is None or a < area:
                best, area = i, a
    return best


def main(a):
    out = Path(a.out)
    classes = read_classes(a.classes or out / "classes_postes.txt")
    imgs = pick_scene_images(out, a.scene, a.limit)
    if not imgs:
        raise SystemExit("No hay fotos visuales para esa escena.")
    lab_dir = out / "labels_postes"
    lab_dir.mkdir(exist_ok=True)
    rev_file = lab_dir / "reviewed.txt"
    reviewed = set(rev_file.read_text().split()) if rev_file.exists() else set()
    print(f"Clases: {list(enumerate(classes, 1))} | fotos: {len(imgs)}")

    st = {"i": next((k for k, (s, _) in enumerate(imgs) if s not in reviewed), 0), "boxes": [], "drag": None,
          "cur": None, "cls": 0, "scale": 1.0, "W": 1, "H": 1, "t0": time.time()}
    times = {}
    win = "Etiquetado multiclase (1-9 clase, d sig, a ant, u deshacer, q salir)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def load(i):
        stem, path = imgs[i]
        img = cv2.imread(path)
        H, W = img.shape[:2]
        st["scale"] = a.max_side / max(W, H)
        st["W"], st["H"] = W, H
        disp = cv2.resize(img, None, fx=st["scale"], fy=st["scale"], interpolation=cv2.INTER_AREA)
        lab = read_labels(lab_dir / f"{stem}.txt")
        st["boxes"] = [(*[v * st["scale"] for v in norm_to_xyxy(b[1:], W, H)], b[0]) for b in lab]
        st["t0"] = time.time()
        return disp

    def save(i):
        stem = imgs[i][0]
        W, H, s = st["W"], st["H"], st["scale"]
        write_labels(lab_dir / f"{stem}.txt",
                     [(c, *xyxy_to_norm((x1 / s, y1 / s, x2 / s, y2 / s), W, H)) for x1, y1, x2, y2, c in st["boxes"]])
        reviewed.add(stem)
        rev_file.write_text("\n".join(sorted(reviewed)))
        times[stem] = times.get(stem, 0) + (time.time() - st["t0"])

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["drag"], st["cur"] = (x, y), (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["cur"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st["drag"]:
            x0, y0 = st["drag"]
            if abs(x - x0) > 4 and abs(y - y0) > 4:
                st["boxes"].append((min(x0, x), min(y0, y), max(x0, x), max(y0, y), st["cls"]))
            st["drag"] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            k = box_at(st["boxes"], x, y)
            if k is not None:
                st["boxes"].pop(k)

    cv2.setMouseCallback(win, on_mouse)
    img = load(st["i"])
    while True:
        vis = img.copy()
        for x1, y1, x2, y2, c in st["boxes"]:
            col = COLORS[c % len(COLORS)]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, classes[c], (int(x1), max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if st["drag"] and st["cur"]:
            cv2.rectangle(vis, st["drag"], st["cur"], COLORS[st["cls"] % len(COLORS)], 1)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
        menu = "  ".join(f"[{k + 1}]{n}" + ("*" if k == st["cls"] else "") for k, n in enumerate(classes[:9]))
        cv2.putText(vis, f"{st['i'] + 1}/{len(imgs)} {imgs[st['i']][0]} | cajas {len(st['boxes'])} | {menu}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        ch = k & 0xFF
        if ord("1") <= ch <= ord("9") and ch - ord("1") < len(classes):
            st["cls"] = ch - ord("1")
        elif ch in (ord("d"), 32) or k == 2555904:
            save(st["i"]); st["i"] = min(st["i"] + 1, len(imgs) - 1); img = load(st["i"])
        elif ch == ord("a") or k == 2424832:
            save(st["i"]); st["i"] = max(st["i"] - 1, 0); img = load(st["i"])
        elif ch == ord("u") and st["boxes"]:
            st["boxes"].pop()
        elif ch == ord("c"):
            st["boxes"] = []
        elif ch in (ord("q"), 27):
            save(st["i"]); break
    cv2.destroyAllWindows()
    if times:
        tot = sum(times.values())
        print(f"Fotos guardadas: {len(reviewed)}/{len(imgs)} | tiempo en esta sesión: {tot / 60:.1f} min "
              f"({tot / len(times):.0f} s por foto en promedio)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--scene", type=int, default=9)
    ap.add_argument("--limit", type=int, default=8, help="cuántas fotos de la escena etiquetar (repartidas parejo)")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--max-side", type=int, default=1200, help="lado mayor de la imagen en pantalla")
    main(ap.parse_args())
