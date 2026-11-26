# PoC — Detección de Puntos Calientes en Imágenes Térmicas (ASICAI)

Prueba de concepto del sistema de IA para detección temprana de anomalías
térmicas (puntos calientes) en imágenes capturadas por drones, como parte
de la pasantía de integración de IA con los drones termográficos de ASICAI.

## Qué hace este sistema

Pipeline completo, reproducible y documentado:

1. **Ingesta de imagen térmica** (actualmente: escenas sintéticas con
   ground truth conocido; listo para recibir imágenes reales de ASICAI).
2. **Detección de puntos calientes** mediante visión por computador
   clásica (umbral adaptativo + componentes conectados).
3. **Clasificación de severidad** (alta / media / baja) según intensidad
   relativa de cada punto respecto al resto de la escena.
4. **Evaluación cuantitativa** (precisión, recall, F1) contra ground truth.
5. **Reporte de salida**: imágenes anotadas + CSV + métricas en JSON.

## Por qué CV clásica y no deep learning (todavía)

Se decidió construir primero un **detector baseline con visión por
computador clásica** (sin redes neuronales) por tres razones:

- No requiere GPU ni un dataset grande etiquetado para funcionar.
- Permite validar el resto del pipeline (ingesta → severidad → reporte)
  de forma rápida y confiable.
- Es la práctica estándar en proyectos de ML: tener un baseline simple
  antes de invertir en un modelo más complejo (YOLO).

El módulo de detección (`src/hotspot_detector.py`) está aislado
justamente para que, en la siguiente fase, se pueda **reemplazar o
combinar con un modelo YOLO ajustado (fine-tuned)** sin tocar el resto
del pipeline. Ver `docs/NEXT_STEPS.md`.

## Cómo correrlo

```bash
pip install -r requirements.txt
python run_demo.py
```

Esto genera:
- `data/sample_images/` — escenas térmicas sintéticas + `ground_truth.json`
- `data/output/annotated_*.png` — imágenes con los hotspots detectados marcados
- `data/output/detecciones.csv` — listado de todas las detecciones
- `data/output/metrics.json` — precisión / recall / F1 del detector

## Notebook de experimentos

`experimentos.ipynb` documenta el funcionamiento del detector y los experimentos
con sus parámetros (umbral, área mínima), con gráficos y celdas para escribir
conclusiones. Ábrelo en VS Code (requiere `pip install -r requirements.txt`).

## Resultados actuales (validación sobre datos sintéticos)

Ver `docs/RESULTADOS.md` para el detalle y la interpretación correcta
de estas métricas (importante: son sobre datos sintéticos, no reales).

## Estructura del repositorio

```
asicai-poc/
├── README.md
├── requirements.txt
├── run_demo.py                  ← orquesta todo el pipeline
├── experimentos.ipynb           ← pruebas documentadas con gráficos
├── src/
│   ├── thermal_simulator.py     ← genera escenas sintéticas + ground truth
│   ├── hotspot_detector.py      ← detector baseline (CV clásica)
│   └── evaluate.py              ← cálculo de métricas
├── data/
│   ├── sample_images/           ← imágenes de entrada (sintéticas por ahora)
│   └── output/                  ← resultados: anotadas, CSV, métricas
└── docs/
    ├── ARCHITECTURE.md
    ├── RESULTADOS.md
    └── NEXT_STEPS.md
```

## Próximo paso inmediato

Reemplazar `thermal_simulator.py` por un loader de imágenes reales de
ASICAI en cuanto estén disponibles (ver `docs/NEXT_STEPS.md`), y usarlas
para (a) recalibrar los umbrales de severidad con casos reales y
(b) preparar el fine-tuning de un modelo YOLO como siguiente iteración.
