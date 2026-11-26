"""Prueba de humo con fotos sintéticas nombradas como DJI. Correr:  python tests/test_pipeline.py"""
import sys, tempfile, shutil, subprocess, os, stat
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from real_data import build_inventory, load_thermal
from real_baseline import detect_simple
import real_baseline, make_yolo_dataset
from eval_real import match, evaluate
from review_labels import box_at, pick_per_group
from yolo_io import iou_xywhn, read_labels, write_labels

rng = np.random.default_rng(0)

def fake_scene(n):
    a = rng.normal(30, 2, (512, 640)).astype(np.float32)
    gt = []
    for _ in range(n):
        cx, cy = int(rng.integers(60, 580)), int(rng.integers(60, 450))
        yy, xx = np.mgrid[:512, :640]
        a += 60 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 8 ** 2))
        gt.append((cx, cy))
    return a, gt

tmp = Path(tempfile.mkdtemp())
photos = tmp / "fotos"; photos.mkdir()
t0 = datetime(2025, 5, 28, 13, 6, 16)
scenes = {}
for i in range(12):
    dt = t0 + timedelta(seconds=(i // 3) * 600 + (i % 3) * 5)   # 4 escenas de 3 fotos
    arr, gt = fake_scene(1 + i % 2)
    u8 = np.clip((arr - arr.min()) / (arr.max() - arr.min()) * 255, 0, 255).astype(np.uint8)
    stem = f"DJI_{dt:%Y%m%d%H%M%S}_{i+1:04d}"
    cv2.imwrite(str(photos / f"{stem}_T.JPG"), cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO))
    cv2.imwrite(str(photos / f"{stem}_V.JPG"), np.zeros((300, 400, 3), np.uint8))
    scenes[f"{dt:%Y%m%d%H%M%S}_{i+1:04d}"] = gt

# 1) inventario y grupos
rows = build_inventory(photos)
assert len(rows) == 12 and len({r["group"] for r in rows}) == 4, (len(rows), {r["group"] for r in rows})
assert all(r["v_path"] for r in rows)

# 2) detector sobre array con 2 puntos
arr, gt = fake_scene(2)
dets, thr = detect_simple(arr)
assert len(dets) == 2, dets
assert detect_simple(np.full((512, 640), 5, np.float32))[0] == []

# 3) SDK falso: prueba el cableado de subprocess
# SDK falso (compatible con Windows y Linux): un .py de ayuda + un lanzador .bat / ejecutable
helper = tmp / "fake_irp.py"
helper.write_text("import sys, numpy as np\n"
                  "a=sys.argv; out=a[a.index('--output')+1]\n"
                  "np.full((512,640),42.5,np.float32).tofile(out)\n")
if os.name == "nt":
    fake = tmp / "dji_irp.bat"
    fake.write_text(f'@echo off\r\n"{sys.executable}" "{helper}" %*\r\n')
else:
    fake = tmp / "dji_irp"
    fake.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{helper}" "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
t, kind = load_thermal(rows[0]["t_path"], sdk_exe=str(fake), cache_dir=tmp / "c")
assert kind == "celsius" and t.shape == (512, 640) and abs(t[0, 0] - 42.5) < 1e-6
t2, k2 = load_thermal(rows[0]["t_path"], cache_dir=tmp / "c")
assert k2 == "celsius"          # con caché de °C previa, siempre se prefiere
t3, k3 = load_thermal(rows[1]["t_path"], cache_dir=tmp / "c2")
assert k3 == "proxy"
# TIFF estilo dji-thermal-cli (<stem>_temp_c.tiff) con un píxel "sin dato"
tif_dir = tmp / "tiffs" / "sub"; tif_dir.mkdir(parents=True)
tt = np.full((512, 640), 35.0, np.float32); tt[10, 10] = -273.15; tt[200:210, 300:310] = 80.0
stem1 = Path(rows[2]["t_path"]).stem
cv2.imwrite(str(tif_dir / f"{stem1}_temp_c.tiff"), tt)
t4, k4 = load_thermal(rows[2]["t_path"], temp_dir=tmp / "tiffs")
assert k4 == "celsius" and abs(t4[250, 250] - 35.0) < 1e-4 and abs(t4[205, 305] - 80.0) < 1e-4 and t4[10, 10] > 0

# 4) pipeline completo en modo proxy
out = tmp / "out"
class A: pass
a = A(); a.root = str(photos); a.out = str(out); a.sdk = None; a.percentile = 97.5; a.std_k = 2.2
a.min_area = 20; a.max_area_frac = None; a.gap_s = 90; a.fixed_range = None
real_baseline.run(a)
assert (out / "index.csv").exists() and len(list((out / "render").glob("*.jpg"))) == 12
assert len(list((out / "labels_pre").glob("*.txt"))) == 12

# 5) "revisión": GT = copia de las pre-etiquetas en 8 fotos (simula revisión manual)
stems = sorted(p.stem for p in (out / "render").glob("*.jpg"))[:8]
(out / "labels_reviewed").mkdir()
for s in stems:
    shutil.copy(out / "labels_pre" / f"{s}.txt", out / "labels_reviewed" / f"{s}.txt")
(out / "reviewed.txt").write_text("\n".join(stems))
make_yolo_dataset.main(str(out), 0.25, 0)
val = (out / "val_stems.txt").read_text().split()
assert val and (out / "yolo_dataset" / "dataset.yaml").exists()
tr = {p.stem for p in (out / "yolo_dataset/images/train").glob("*.jpg")}
inv = {r["stem"]: r["group"] for r in __import__("real_data").load_inventory(out / "index.csv")}
assert not ({inv[s] for s in val} & {inv[s] for s in tr}), "FUGA: misma escena en train y val"

# 6) evaluación: baseline vs GT idéntico => F1=1; con una caja inventada => FP
m = evaluate(out / "labels_reviewed", out / "labels_pre", val, 0.5)
assert m["f1"] == 1.0, m
assert match([(0.5, 0.5, 0.1, 0.1)], [(0.9, 0.9, 0.1, 0.1)], 0.1) == (0, 1, 1)
assert abs(iou_xywhn((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) - 1) < 1e-9
assert box_at([(0, 0, 100, 100), (10, 10, 20, 20)], 15, 15) == 1 and box_at([(0, 0, 5, 5)], 50, 50) is None
import real_tools
arrs = real_tools.cargar_arrays(out, val)
r = real_tools.eval_baseline(arrs, out / "labels_reviewed", val, iou=0.5)
assert r["f1"] == 1.0, r
assert len(real_tools.errores_por_foto(arrs, out / "labels_reviewed", val)) == len(val)
g_of = {f"s{i:02d}": str(i // 10) for i in range(25)}      # 3 escenas (10, 10, 5 fotos)
sel = pick_per_group(list(g_of), g_of, 3)
assert len(sel) == 9 and {g_of[s] for s in sel} == {"0", "1", "2"}, sel
assert pick_per_group(["a", "b"], {"a": "0", "b": "0"}, 3) == ["a", "b"]
import scene_catalog
scene_catalog.main(out)
assert (out / "catalogo" / "resumen.jpg").exists() and (out / "catalogo" / "catalogo_escenas.csv").exists()
import transfer_insplad as ti
_x, _r, _px, _py = ti.letterbox(np.zeros((3000, 4000, 3), np.uint8), 640)
_o = np.zeros((21, 2), np.float32); _g = [1000 * _r + _px, 600 * _r + _py, 1800 * _r + _px, 1400 * _r + _py]
_o[:4, 0] = [(_g[0] + _g[2]) / 2, (_g[1] + _g[3]) / 2, _g[2] - _g[0], _g[3] - _g[1]]; _o[4 + 5, 0] = 0.9
_d = ti.decode(_o, 640, _r, _px, _py, 4000, 3000)
assert len(_d) == 1 and all(abs(u - v) < 1 for u, v in zip(_d[0]["bbox"], (1000, 600, 1800, 1400)))
import zero_shot_owl as zo
_d = zo.postprocess([[1000, 600, 1800, 1400], [1010, 610, 1810, 1410], [500, 2900, 900, 3600]], [0.4, 0.3, 0.2], [0, 0, 1], 4000, 3000)
assert len(_d) == 2 and _d[1][2][3] == 3000
_b, _s, _l = zo.decode_owl(np.array([[-5., 3.0], [-5., -4.0]], np.float32), np.array([[0.5, 0.4, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]], np.float32), 4000, 0.10)
assert len(_b) == 1 and _l[0] == 1 and np.allclose(_b[0], [1600, 1200, 2400, 2000])
shutil.rmtree(tmp)
print("TODO OK")
