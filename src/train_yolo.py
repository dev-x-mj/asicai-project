"""
Fine-tuning de YOLO (Ultralytics) con el dataset revisado, y predicción sobre validación.

  pip install ultralytics
  python src/train_yolo.py --out data/real_private --epochs 100 --device cpu
  python src/train_yolo.py --out data/real_private --predict runs/asicai/hotspot/weights/best.pt

Notas:
 - 'yolov8n.pt' se descarga de Internet la primera vez (modelo preentrenado en COCO, fotos normales).
 - Desactivo el cambio de tono (hsv_h=0) y de saturación: en falso color el COLOR significa temperatura.
 - Si no tienes GPU, usa --device cpu y pocas épocas para probar. NO subas las fotos a Colab/otra nube
   sin permiso escrito de ASICAI (son de sus clientes).
"""
import argparse
from pathlib import Path


def train(a):
    from ultralytics import YOLO
    ds = Path(a.out) / "yolo_dataset" / "dataset.yaml"
    model = YOLO(a.weights)
    model.train(data=str(ds), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=a.project, name="hotspot", seed=0, patience=25, exist_ok=True,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.1, fliplr=0.5, flipud=0.0, mosaic=1.0)
    print("Mejor modelo:", Path(a.project) / "hotspot" / "weights" / "best.pt")


def predict(a):
    from ultralytics import YOLO
    out = Path(a.out)
    stems = (out / "val_stems.txt").read_text().split()
    pred_dir = out / "pred_yolo"
    pred_dir.mkdir(exist_ok=True)
    model = YOLO(a.predict)
    for s in stems:
        res = model.predict(str(out / "render" / f"{s}.jpg"), conf=a.conf, imgsz=a.imgsz,
                            device=a.device, verbose=False)[0]
        with open(pred_dir / f"{s}.txt", "w") as f:
            for b, c in zip(res.boxes.xywhn.tolist(), res.boxes.conf.tolist()):
                f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {c:.4f}\n")
    print("Predicciones en:", pred_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_private")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="runs/asicai")
    ap.add_argument("--predict", default=None, help="ruta a best.pt para predecir sobre val")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    predict(a) if a.predict else train(a)
