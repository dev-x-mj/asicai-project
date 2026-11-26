# Arquitectura del sistema

## Diagrama de flujo

```
 [Imagen térmica]
        │
        ▼
 ┌─────────────────────┐
 │ Preprocesamiento     │   (escala de grises / matriz de intensidad)
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ Detector de hotspots │   src/hotspot_detector.py
 │ (CV clásica, hoy)    │   → reemplazable por YOLO fine-tuned (futuro)
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ Clasificación de     │   severidad: alta / media / baja
 │ severidad            │
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ Evaluación           │   src/evaluate.py (solo si hay ground truth)
 │ (precisión/recall)   │
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ Reporte de salida    │   imágenes anotadas + CSV + métricas JSON
 └─────────────────────┘
```

## Decisiones de diseño clave

**Modularidad**: cada etapa es una función independiente con una
responsabilidad única. Esto permite que, por ejemplo, el detector de
hotspots se reemplace por un modelo YOLO sin modificar el resto del
sistema (evaluación, reporte, etc. siguen funcionando igual).

**Umbral adaptativo, no fijo**: el detector usa percentil + media/desvío
estándar de la propia escena en vez de un valor de temperatura fijo.
Esto es necesario porque la temperatura ambiente varía entre inspecciones
(hora del día, clima, tipo de activo), y un umbral fijo no generalizaría.

**Separación entre simulación y detección**: `thermal_simulator.py` es
el único módulo que se reemplazará al integrar datos reales de ASICAI.
El resto del pipeline (detección, evaluación, reporte) no depende de que
los datos sean sintéticos o reales — solo espera una imagen en escala de
grises como entrada.

**Extensibilidad a otros casos de uso con drones**: el pipeline no asume
nada específico de termografía más allá de la entrada (imagen en escala
de grises). La misma arquitectura (ingesta → detección → clasificación →
evaluación → reporte) aplica a otros casos de ASICAI con drones
(inspección de líneas eléctricas, oleoductos, mapeo), cambiando el
detector específico de cada dominio.
