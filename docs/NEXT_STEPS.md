# Próximos pasos

## Inmediato (requiere a ASICAI)

1. **Obtener un set inicial de imágenes térmicas reales** (aunque sean
   pocas, 20-50 imágenes ya permiten recalibrar el detector).
2. Confirmar **modelo de dron/cámara térmica** usado, para saber si se
   puede extraer la matriz radiométrica real (temperatura por píxel) en
   vez de trabajar solo con la imagen visual de 8 bits.
3. Si existen, levantar **casos históricos donde una falla fue
   confirmada** — son el ground truth más valioso que existe.

## Siguiente iteración técnica

- Recalibrar los umbrales de severidad (`hotspot_detector.py`) con
  datos reales en vez de los valores sintéticos actuales.
- Fine-tuning de un modelo YOLO (Ultralytics) sobre las imágenes reales
  etiquetadas (con Roboflow u otra herramienta de anotación), para
  reemplazar o complementar el detector clásico — especialmente útil
  cuando los hotspots tienen formas irregulares que el umbral simple no
  captura bien (ej. patrones de corrosión, fugas).
- Nota técnica: el entorno de desarrollo actual no cuenta con una
  instalación funcional de PyTorch/GPU; antes de esta fase hay que
  asegurar un entorno con esos recursos (local de ASICAI o la nube).

## Extensión a otros casos de uso (mediano plazo)

La arquitectura modular (ver `ARCHITECTURE.md`) está pensada para que
el mismo pipeline sirva de base a otros casos de ASICAI con drones:
inspección de líneas eléctricas, seguimiento de oleoductos, mapeo. Se
sugiere definir junto a ASICAI cuál sería el segundo caso de uso a
abordar una vez el de termografía esté validado con datos reales.
