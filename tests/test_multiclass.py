"""python tests/test_multiclass.py  (prueba de humo del etiquetado multiclase y su evaluación)"""
import csv, sys, tempfile, shutil
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import labels_multi as L
from review_multiclass import box_at
import eval_owl_vs_labels as E

tmp = Path(tempfile.mkdtemp()); out = tmp / "out"; (out / "labels_postes").mkdir(parents=True)
# 10 fotos visuales falsas de 400x300 en la escena 9
rows = []
for i in range(10):
    p = tmp / f"v{i}.jpg"; cv2.imwrite(str(p), np.zeros((300, 400, 3), np.uint8))
    rows.append({"stem": f"s{i:02d}", "datetime": f"2025-05-28T12:00:{i:02d}", "group": 9, "t_path": "", "v_path": str(p)})
rows.append({"stem": "otra", "datetime": "2025-05-28T11:00:00", "group": 3, "t_path": "", "v_path": str(tmp / "v0.jpg")})
with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

sel = L.pick_scene_images(out, 9, 4)
assert len(sel) == 4 and sel[0][0] == "s00" and sel[-1][0] == "s09", sel      # repartidas parejo, solo escena 9
assert len(L.pick_scene_images(out, 9)) == 10 and L.pick_scene_images(out, 5) == []

classes = L.read_classes(out / "classes_postes.txt"); assert classes[:3] == ["poste", "cruceta", "aislador"]
# etiquetas: 1 poste + 2 aisladores por foto en 2 fotos
gt = {"s00": [(0, 100, 50, 140, 250), (2, 200, 60, 230, 90), (2, 250, 60, 280, 90)],
      "s09": [(0, 100, 50, 140, 250), (2, 200, 60, 230, 90)]}
for stem, bs in gt.items():
    L.write_labels(out / "labels_postes" / f"{stem}.txt", [(c, *L.xyxy_to_norm((a, b, c_, d), 400, 300)) for c, a, b, c_, d in bs])
(out / "labels_postes" / "reviewed.txt").write_text("s00\ns09")
back = L.read_labels(out / "labels_postes" / "s00.txt"); assert len(back) == 3 and back[0][0] == 0

# detecciones: poste correcto (0.6), aislador correcto (0.4), aislador correcto débil (0.12), 1 aislador falso (0.3), 1 poste falso (0.11)
dets = [("s00", "utility pole", .6, (100, 50, 140, 250)), ("s00", "insulator", .4, (200, 60, 230, 90)),
        ("s00", "insulator", .12, (250, 60, 280, 90)), ("s00", "insulator", .3, (10, 10, 40, 40)),
        ("s09", "utility pole", .11, (300, 200, 340, 290)), ("s09", "utility pole", .7, (100, 50, 140, 250)),
        ("s09", "insulator", .5, (200, 60, 230, 90))]
with open(tmp / "det.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["stem", "clase", "conf", "x1", "y1", "x2", "y2"])
    for s, q, c, b in dets: w.writerow([s, q, c, *b])
res = {(r["clase"], r["umbral"]): r for r in E.evaluate(out, tmp / "det.csv", (0.1, 0.3, 0.5), 0.5)}
p01, p05 = res[("poste", 0.1)], res[("poste", 0.5)]
assert (p01["TP"], p01["FP"], p01["FN"]) == (2, 1, 0), p01               # a 0.1 entra el falso (0.11)
assert (p05["TP"], p05["FP"], p05["FN"]) == (2, 0, 0) and p05["f1"] == 1.0
a01, a03, a05 = res[("aislador", 0.1)], res[("aislador", 0.3)], res[("aislador", 0.5)]
assert (a01["TP"], a01["FP"], a01["FN"]) == (3, 1, 0), a01
assert (a03["TP"], a03["FP"], a03["FN"]) == (2, 1, 1), a03
assert (a05["TP"], a05["FP"], a05["FN"]) == (1, 0, 2), a05
assert res[("cruceta", 0.1)]["cajas_reales"] == 0 and res[("cruceta", 0.1)]["TP"] == 0   # mapeada pero sin etiquetas
(out / "classes_postes.txt").write_text("poste\nretenida\n")
assert ("retenida", 0.1) not in {(r["clase"], r["umbral"]) for r in E.evaluate(out, tmp / "det.csv", (0.1,), 0.5)}  # sin consulta mapeada
assert box_at([(0, 0, 100, 100, 0), (10, 10, 20, 20, 1)], 15, 15) == 1 and box_at([(0, 0, 5, 5, 0)], 50, 50) is None
# zona "dudoso": un aislador falso (0.3) que cae sobre un objeto marcado dudoso NO cuenta como FP
(out / "classes_postes.txt").write_text("poste\ncruceta\naislador\ntransformador\ndudoso\n")
L.write_labels(out / "labels_postes" / "s00.txt", [(c, *L.xyxy_to_norm((a, b, c_, d), 400, 300)) for c, a, b, c_, d in
               gt["s00"] + [(4, 5, 5, 45, 45)]])
r2 = {(r["clase"], r["umbral"]): r for r in E.evaluate(out, tmp / "det.csv", (0.1, 0.3), 0.5)}
assert r2[("aislador", 0.3)]["FP"] == 0 and r2[("aislador", 0.3)]["ignoradas_dudoso"] == 1, r2[("aislador", 0.3)]
assert r2[("aislador", 0.3)]["TP"] == 2 and r2[("aislador", 0.3)]["FN"] == 1
# varias descripciones sobre el mismo objeto -> una sola caja
mm = E.merge_nms([(0.6, L.xyxy_to_norm((100, 50, 140, 250), 400, 300)), (0.4, L.xyxy_to_norm((102, 52, 141, 249), 400, 300)),
                  (0.3, L.xyxy_to_norm((300, 10, 340, 60), 400, 300))])
assert len(mm) == 2
shutil.rmtree(tmp); print("MULTICLASE OK")