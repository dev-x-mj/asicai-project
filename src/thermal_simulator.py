"""
thermal_simulator.py
---------------------
Genera imágenes térmicas SINTÉTICAS con puntos calientes de severidad conocida.

Por qué existe este módulo:
ASICAI aún no ha entregado un set de imágenes térmicas reales (o no hemos
tenido acceso a él en este sprint). Para poder validar el pipeline de
detección de punto a punto, generamos escenas térmicas realistas (ruido de
fondo + "componentes" fríos + hotspots gaussianos de distinta intensidad)
con las coordenadas de cada hotspot GUARDADAS como ground truth.

Esto permite calcular métricas reales (precisión/recall) del detector,
en vez de solo mostrar "se ve bien". En cuanto existan imágenes reales de
ASICAI, este módulo se reemplaza por un loader real (ver data_loader.py)
y el resto del pipeline (detector, evaluación, reporte) no cambia.
"""
import numpy as np
import json
import os
import cv2

RNG = np.random.default_rng(42)


def _gaussian_blob(size, cy, cx, sigma, amplitude):
    y, x = np.ogrid[:size[0], :size[1]]
    g = amplitude * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2)))
    return g


def generate_scene(width=480, height=320, n_hotspots=3, seed=None):
    """Genera una escena térmica sintética (valores 0-255 ~ 'temperatura relativa').

    Retorna: (imagen uint8, lista_ground_truth)
    lista_ground_truth: [{"cx","cy","radius_px","severity","delta_t_proxy"}]
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    base = rng.normal(loc=80, scale=6, size=(height, width))  # fondo "ambiente"

    # Simula un par de componentes industriales (conectores/tuberías) a temperatura
    # cercana al ambiente (no deben confundirse con hotspots reales)
    for _ in range(rng.integers(2, 4)):
        cy, cx = rng.integers(40, height - 40), rng.integers(40, width - 40)
        w, h = rng.integers(30, 70), rng.integers(15, 35)
        comp_temp = float(rng.uniform(78, 84))
        cv2.rectangle(base, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), comp_temp, -1)

    gt = []
    severities = ["alta", "media", "baja"]
    for i in range(n_hotspots):
        cy = rng.integers(30, height - 30)
        cx = rng.integers(30, width - 30)
        sigma = rng.uniform(6, 14)
        amplitude = rng.uniform(55, 150)  # qué tan "caliente" es el punto
        base = base + _gaussian_blob((height, width), cy, cx, sigma, amplitude)

        if amplitude > 100:
            sev = "alta"
        elif amplitude > 75:
            sev = "media"
        else:
            sev = "baja"

        gt.append({
            "id": i,
            "cx": int(cx), "cy": int(cy),
            "radius_px": round(float(sigma * 2), 1),
            "severity": sev,
            "delta_t_proxy": round(float(amplitude), 1),
        })

    img = np.clip(base, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, gt


def save_dataset(out_dir, n_images=8):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for idx in range(n_images):
        n_hot = int(RNG.integers(1, 4))
        img, gt = generate_scene(n_hotspots=n_hot, seed=1000 + idx)
        fname = f"scene_{idx:03d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        manifest.append({"file": fname, "ground_truth": gt})

    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    m = save_dataset("/home/claude/asicai-poc/data/sample_images", n_images=8)
    print(f"Generadas {len(m)} escenas térmicas sintéticas con ground truth.")
