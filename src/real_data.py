"""
Manejo de fotos reales del DJI Mavic 3T (archivos DJI_AAAAMMDDhhmmss_NNNN_T.JPG / _V.JPG).

  _T = térmica (640x512)      _V = visual (cámara RGB)

IMPORTANTE (confidencialidad): todo corre LOCAL. Ninguna foto sale de tu PC.
"""
import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # silencia avisos inofensivos de tags TIFF de DJI
import cv2
import numpy as np
from PIL import Image

PAT = re.compile(r"^DJI_(\d{14})_(\d{4})_([A-Z])\.(jpg|jpeg)$", re.I)
THERMAL_SHAPE = (512, 640)  # (alto, ancho) de la cámara térmica del Mavic 3T


# ----------------------------------------------------------------- inventario
def parse_name(path):
    """DJI_20250528130616_0001_T.JPG -> (datetime, 1, 'T') o None si no cumple el patrón."""
    m = PAT.match(Path(path).name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), int(m.group(2)), m.group(3).upper()


def build_inventory(root, gap_s=90):
    """
    Recorre 'root', empareja T con V y asigna un 'group' por cercanía temporal.

    group: las fotos tomadas con menos de gap_s segundos de diferencia son casi la
    misma escena. Al dividir train/val hay que separar por GRUPO (no por foto),
    si no el modelo "ve en entrenamiento" la misma escena que luego evalúa (fuga de datos).
    """
    items = {}
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        r = parse_name(p)
        if r is None:
            continue
        dt, seq, kind = r
        key = (str(p.parent), dt, seq)
        d = items.setdefault(key, {"dt": dt, "seq": seq, "folder": str(p.parent)})
        d[kind] = str(p)

    rows = sorted((v for v in items.values() if "T" in v), key=lambda v: v["dt"])
    grupo, prev = -1, None
    out = []
    for v in rows:
        if prev is None or (v["dt"] - prev).total_seconds() > gap_s:
            grupo += 1
        prev = v["dt"]
        with Image.open(v["T"]) as im:
            w, h = im.size
        out.append({
            "stem": f"{v['dt']:%Y%m%d%H%M%S}_{v['seq']:04d}",
            "datetime": v["dt"].isoformat(),
            "group": grupo,
            "t_path": v["T"],
            "v_path": v.get("V", ""),
            "width": w, "height": h,
            "size_kb": round(Path(v["T"]).stat().st_size / 1024),
        })
    return out


def save_inventory(rows, csv_path):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_inventory(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ lectura térmica
def index_temp_tiffs(temp_dir):
    """{stem_de_la_foto_T: ruta} para los *_temp_c.tif(f) que genera dji-thermal-cli."""
    out = {}
    if temp_dir:
        for p in Path(temp_dir).rglob("*"):
            n = p.name
            low = n.lower()
            if low.endswith(("_temp_c.tiff", "_temp_c.tif")):
                out[n[: low.rindex("_temp_c")]] = p
    return out


def _clean(arr):
    """Los píxeles sin dato salen como ~ -273 °C: se sustituyen para no arruinar percentiles."""
    arr = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(arr) | (arr < -270)
    if bad.any():
        good = arr[~bad]
        arr = arr.copy()
        arr[bad] = float(np.median(good)) if good.size else 0.0
    return arr


def load_thermal(t_path, sdk_exe=None, cache_dir=None, sdk_extra_args=None, temp_dir=None):
    """
    Devuelve (matriz float32 HxW, tipo).
      tipo = "celsius" -> temperaturas reales (R-JPEG). Tres orígenes, en este orden:
               1) caché .npy previa de °C   2) TIFF de dji-thermal-cli en temp_dir
               3) DJI Thermal SDK oficial (sdk_exe)
      tipo = "proxy"   -> SOLO brillo de la imagen en color. NO es temperatura.

    Por qué importa: el _T.JPG que ves es una imagen en falso color (8 bits). La
    temperatura real (16 bits) va escondida dentro del archivo. Además tus fotos usan
    paletas distintas (unas tipo hierro/fuego, otras tipo arcoíris): pasar a gris una
    paleta arcoíris NO conserva el orden frío->caliente. Por eso 'proxy' es solo para pruebas.
    """
    t_path = Path(t_path)
    cache_c = cache_p = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_c = Path(cache_dir) / f"{t_path.stem}_c.npy"
        cache_p = Path(cache_dir) / f"{t_path.stem}_p.npy"
        if cache_c.exists():                      # si ya hay °C, siempre se prefieren
            return np.load(cache_c), "celsius"

    tiffs = index_temp_tiffs(temp_dir) if temp_dir else {}
    if t_path.stem in tiffs:
        arr = cv2.imread(str(tiffs[t_path.stem]), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.ndim != 2:
            raise RuntimeError(f"No pude leer el TIFF de temperatura: {tiffs[t_path.stem]}")
        arr, kind = _clean(arr), "celsius"
    elif sdk_exe:
        arr, kind = _clean(_load_with_sdk(t_path, sdk_exe, sdk_extra_args or [])), "celsius"
    else:
        if cache_p is not None and cache_p.exists():
            return np.load(cache_p), "proxy"
        img = cv2.imread(str(t_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(t_path)
        arr, kind = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), "proxy"

    if cache_dir:
        np.save(cache_c if kind == "celsius" else cache_p, arr)
    return arr, kind


def _load_with_sdk(t_path, sdk_exe, extra_args):
    """
    Llama al ejecutable 'dji_irp' del DJI Thermal SDK y lee la matriz float32.
    Sintaxis tomada de ejemplos públicos del SDK (repositorio dji-thermal-sdk-docker):
      dji_irp -a measure --measurefmt float32 --distance 25 --humidity 77 --emissivity 0.98
              --reflection 23 --source FOTO_T.JPG --output salida.raw
    Verifica con `dji_irp --help` en tu versión. Parámetros físicos opcionales se pasan en
    extra_args, p. ej. ["--distance","5","--emissivity","0.95"].
    """
    import tempfile
    sdk_exe = Path(sdk_exe)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "t.raw"
        cmd = [str(sdk_exe), "-a", "measure", "--measurefmt", "float32", *extra_args,
               "--source", str(t_path), "--output", str(out)]
        r = subprocess.run(cmd, cwd=str(sdk_exe.parent), capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"El SDK falló.\nCMD: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        raw = np.fromfile(out, dtype=np.float32)
    h, w = THERMAL_SHAPE
    if raw.size != h * w:
        raise RuntimeError(f"Tamaño inesperado: {raw.size} valores (esperaba {h*w}). Revisa --measurefmt y la resolución.")
    return raw.reshape(h, w)


# ---------------------------------------------------------------- visualización
def render_uniform(arr, lo=None, hi=None):
    """
    Convierte la matriz a imagen BGR con UNA sola paleta (inferno) para todas las fotos.
    - Sin lo/hi: normaliza por imagen (percentiles 1-99.8). Cómodo para ver, pero oculta
      temperaturas absolutas (una escena "fría" se ve igual de intensa).
    - Con lo/hi (p.ej. 20 y 120 °C): rango fijo, comparable entre fotos.
    """
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1)) if lo is None else lo
    hi = float(np.percentile(a, 99.8)) if hi is None else hi
    if hi - lo < 1e-6:
        hi = lo + 1.0
    u8 = np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
