"""
Arma el dataset YOLO SOLO con fotos revisadas a mano, separando train/val por ESCENA.

  python src/make_yolo_dataset.py --out data/real_private --val-frac 0.25

Crea  <out>/yolo_dataset/{images,labels}/{train,val}, dataset.yaml y val_stems.txt
"""
import argparse
import random
import shutil
from pathlib import Path

from real_data import load_inventory


def main(out, val_frac, seed):
    out = Path(out)
    rev_file = out / "reviewed.txt"
    if not rev_file.exists():
        raise SystemExit("No hay reviewed.txt. Revisa fotos con review_labels.py primero.")
    reviewed = set(rev_file.read_text().split())
    inv = [r for r in load_inventory(out / "index.csv") if r["stem"] in reviewed]
    if len(inv) < 8:
        print(f"AVISO: solo {len(inv)} fotos revisadas. Con tan pocas, cualquier métrica será muy inestable.")

    groups = sorted({r["group"] for r in inv}, key=int)
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_frac)) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])

    ds = out / "yolo_dataset"
    if ds.exists():
        shutil.rmtree(ds)
    val_stems = []
    for r in inv:
        split = "val" if r["group"] in val_groups else "train"
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "render" / f"{r['stem']}.jpg", ds / "images" / split / f"{r['stem']}.jpg")
        shutil.copy(out / "labels_reviewed" / f"{r['stem']}.txt", ds / "labels" / split / f"{r['stem']}.txt")
        if split == "val":
            val_stems.append(r["stem"])

    (ds / "dataset.yaml").write_text(
        f"path: {ds.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: hotspot\n")
    (out / "val_stems.txt").write_text("\n".join(sorted(val_stems)))
    ntr = len(inv) - len(val_stems)
    print(f"Escenas: {len(groups)} (val: {n_val}) | fotos train: {ntr} | fotos val: {len(val_stems)}")
    print(f"dataset.yaml: {(ds / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.val_frac, a.seed)
