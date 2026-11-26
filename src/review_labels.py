"""
Revisor de etiquetas LOCAL (OpenCV). Sirve para crear el "ground truth" real.

  python src/review_labels.py --out data/real_private            # parte de las cajas del baseline
  python src/review_labels.py --out data/real_private --blind    # parte SIN cajas (recomendado: evita sesgo)
  python src/review_labels.py --out data/real_private --blind --per-group 3   # 3 fotos por escena

Qué es una etiqueta aquí: "zona visiblemente más caliente que su entorno que un técnico
querría revisar". NO es un diagnóstico de falla. La definición de qué cuenta como punto
de interés la debe confirmar un técnico de ASICAI [pendiente].

Controles:
  clic izquierdo + arrastrar : dibujar caja      clic derecho : borrar la caja bajo el cursor
  d / flecha der. / espacio  : guardar y siguiente   a / flecha izq. : guardar y anterior
  c : borrar todas las cajas      r : volver a las cajas iniciales      q / ESC : guardar y salir
Una foto cuenta como "revisada" cuando sales de ella con d/a/q (aunque no tenga cajas: eso es
un negativo válido y vale mucho para entrenar).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from yolo_io import read_labels, write_labels, xywhn_to_xyxy, xyxy_to_xywhn


def temp_at(arr, x, y):
    """Temperatura (°C) de la matriz 'arr' en el píxel (x, y), o None si está fuera o no hay datos."""
    if arr is None:
        return None
    xi, yi = int(x), int(y)
    if 0 <= yi < arr.shape[0] and 0 <= xi < arr.shape[1]:
        return float(arr[yi, xi])
    return None


def find_temp_cache(out, stem):
    """Ruta del .npy de °C de una foto. Los archivos se guardan como DJI_<stem>_T_c.npy (nombre original de la foto térmica)."""
    d = Path(out) / "temp_cache"
    for name in (f"DJI_{stem}_T_c.npy", f"{stem}_c.npy"):
        if (d / name).exists():
            return d / name
    return None


def box_at(boxes_px, x, y):
    """Índice de la caja más pequeña que contiene el punto (x,y), o None."""
    best, best_area = None, None
    for i, (x1, y1, x2, y2) in enumerate(boxes_px):
        if x1 <= x <= x2 and y1 <= y <= y2:
            a = (x2 - x1) * (y2 - y1)
            if best is None or a < best_area:
                best, best_area = i, a
    return best


def pick_per_group(stems, group_of, k):
    """Elige k fotos por escena, repartidas a lo largo de la escena (reproducible)."""
    by = {}
    for s in sorted(stems):
        by.setdefault(group_of[s], []).append(s)
    chosen = []
    for g in sorted(by, key=int):
        lst = by[g]
        if len(lst) <= k:
            chosen += lst
        else:
            chosen += [lst[round(i * (len(lst) - 1) / (k - 1))] if k > 1 else lst[len(lst) // 2] for i in range(k)]
    return sorted(set(chosen))


def main(out, blind, scale, per_group=None):
    out = Path(out)
    stems = sorted(p.stem for p in (out / "render").glob("*.jpg"))
    if per_group:
        from real_data import load_inventory
        group_of = {r["stem"]: r["group"] for r in load_inventory(out / "index.csv")}
        stems = pick_per_group(stems, group_of, per_group)
        print(f"Revisión estratificada: {len(stems)} fotos ({per_group} por escena)")
    if not stems:
        raise SystemExit("No hay imágenes en render/. Corre antes real_baseline.py")
    rev_dir = out / "labels_reviewed"
    rev_dir.mkdir(exist_ok=True)
    rev_file = out / "reviewed.txt"
    reviewed = set(rev_file.read_text().split()) if rev_file.exists() else set()

    state = {"i": next((k for k, s in enumerate(stems) if s not in reviewed), 0),
             "boxes": [], "drag": None, "cur": None, "temp": None, "mouse": None}
    win = "Revision (d=sig, a=ant, clic der=borrar, q=salir)"
    n_t = sum(1 for s in stems if find_temp_cache(out, s))
    print(f"Temperaturas encontradas para {n_t} de {len(stems)} fotos" + ("" if n_t else " -> no se mostrará la temperatura bajo el mouse"))
    cv2.namedWindow(win)

    def initial_boxes(stem, H, W):
        src = rev_dir / f"{stem}.txt"
        if src.exists():
            lab = read_labels(src)
        elif blind:
            lab = []
        else:
            lab = read_labels(out / "labels_pre" / f"{stem}.txt")
        return [xywhn_to_xyxy(b, W, H) for b in lab]

    def load(i):
        stem = stems[i]
        img = cv2.imread(str(out / "render" / f"{stem}.jpg"))
        H, W = img.shape[:2]
        state["boxes"] = initial_boxes(stem, H, W)
        npy = find_temp_cache(out, stem)                      # °C reales (solo si se extrajeron)
        state["temp"] = np.load(npy) if npy else None
        return img

    def save(i, img):
        H, W = img.shape[:2]
        stem = stems[i]
        write_labels(rev_dir / f"{stem}.txt", [xyxy_to_xywhn(b, W, H) for b in state["boxes"]])
        reviewed.add(stem)
        rev_file.write_text("\n".join(sorted(reviewed)))

    def on_mouse(ev, x, y, flags, _):
        x, y = x / scale, y / scale
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y); state["cur"] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["cur"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                state["boxes"].append((min(x0, x), min(y0, y), max(x0, x), max(y0, y)))
            state["drag"] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            k = box_at(state["boxes"], x, y)
            if k is not None:
                state["boxes"].pop(k)

    cv2.setMouseCallback(win, on_mouse)
    img = load(state["i"])
    while True:
        vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for x1, y1, x2, y2 in state["boxes"]:
            cv2.rectangle(vis, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 255, 0), 2)
        if state["drag"] and state["cur"]:
            (x0, y0), (x1, y1) = state["drag"], state["cur"]
            cv2.rectangle(vis, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)), (255, 255, 0), 1)
        i = state["i"]
        txt = f"{i+1}/{len(stems)} {stems[i]} | cajas: {len(state['boxes'])} | revisadas: {len(reviewed)}"
        tt = temp_at(state["temp"], *state["mouse"]) if state["mouse"] else None
        if tt is not None:
            txt += f" | bajo el mouse: {tt:.1f} C (foto: min {state['temp'].min():.1f}, max {state['temp'].max():.1f})"
        cv2.putText(vis, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        ch = k & 0xFF
        if ch in (ord("d"), 32) or k == 2555904:       # siguiente
            save(i, img); state["i"] = min(i + 1, len(stems) - 1); img = load(state["i"])
        elif ch == ord("a") or k == 2424832:            # anterior
            save(i, img); state["i"] = max(i - 1, 0); img = load(state["i"])
        elif ch == ord("c"):
            state["boxes"] = []
        elif ch == ord("r"):
            H, W = img.shape[:2]
            src = out / "labels_pre" / f"{stems[i]}.txt"
            state["boxes"] = [] if blind else [xywhn_to_xyxy(b, W, H) for b in read_labels(src)]
        elif ch in (ord("q"), 27):
            save(i, img); break
    cv2.destroyAllWindows()
    print(f"Revisadas: {len(reviewed)}/{len(stems)}  ->  {rev_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--per-group", type=int, default=None,
                    help="revisar solo K fotos por escena (cubre todas las escenas; p.ej. 3)")
    a = ap.parse_args()
    main(a.out, a.blind, a.scale, a.per_group)