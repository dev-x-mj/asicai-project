"""
Catálogo de escenas: ¿qué hay realmente en las 170 fotos? (equipo, postes, líneas, etc.)

  python src/scene_catalog.py --out data/real_private

Genera en <out>/catalogo/ (LOCAL y CONFIDENCIAL):
  resumen.jpg          una miniatura por escena, numerada (vista rápida de todo el conjunto)
  escena_XX.jpg        hasta 8 fotos de esa escena (cámara visual _V; si no hay, la térmica)
  catalogo_escenas.csv una fila por escena con la columna tipo_contenido VACÍA para que la llenes
Ejemplos de valores para tipo_contenido: tablero, motor, transformador, poste, linea, torre, otro.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from real_data import load_inventory


def _thumb(path, w=320):
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((int(w * 0.75), w, 3), np.uint8)
    h = max(1, int(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _pad(img, h, w):
    out = np.zeros((h, w, 3), np.uint8)
    out[:min(h, img.shape[0]), :min(w, img.shape[1])] = img[:h, :w]
    return out


def _grid(imgs, cols):
    h = max(i.shape[0] for i in imgs); w = max(i.shape[1] for i in imgs)
    imgs = [_pad(i, h, w) for i in imgs]
    while len(imgs) % cols:
        imgs.append(np.zeros((h, w, 3), np.uint8))
    return np.vstack([np.hstack(imgs[k:k + cols]) for k in range(0, len(imgs), cols)])


def main(out, max_per_scene=8):
    out = Path(out)
    inv = load_inventory(out / "index.csv")
    cat = out / "catalogo"
    cat.mkdir(exist_ok=True)
    groups = {}
    for r in inv:
        groups.setdefault(int(r["group"]), []).append(r)

    def src(r):                                   # preferir la cámara visual
        v = r.get("v_path")
        return v if v and Path(v).exists() else out / "render" / f"{r['stem']}.jpg"

    rows, firsts = [], []
    for g in sorted(groups):
        lst = sorted(groups[g], key=lambda r: r["datetime"])
        idx = np.linspace(0, len(lst) - 1, min(max_per_scene, len(lst))).round().astype(int)
        thumbs = [_label(_thumb(src(lst[i])), lst[i]["stem"]) for i in idx]
        cv2.imwrite(str(cat / f"escena_{g:02d}.jpg"), _grid(thumbs, 4))
        firsts.append(_label(_thumb(src(lst[0]), 240), f"escena {g:02d} ({len(lst)} fotos)"))
        rows.append({"escena": g, "n_fotos": len(lst), "carpeta": Path(lst[0]["t_path"]).parent.name,
                     "inicio": lst[0]["datetime"][:19], "fin": lst[-1]["datetime"][:19],
                     "primera_foto": lst[0]["stem"], "tipo_contenido": "", "notas": ""})
    cv2.imwrite(str(cat / "resumen.jpg"), _grid(firsts, 5))
    with open(cat / "catalogo_escenas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(groups)} escenas catalogadas en {cat.resolve()}")
    print("Abre resumen.jpg y llena la columna tipo_contenido de catalogo_escenas.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--max-per-scene", type=int, default=8)
    a = ap.parse_args()
    main(a.out, a.max_per_scene)
