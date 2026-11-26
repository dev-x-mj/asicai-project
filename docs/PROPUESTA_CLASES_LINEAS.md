# Propuesta de clases — postes y líneas (para validar con ASICAI)

Estado: **BORRADOR**. Los nombres salen de lo que se aprecia en las fotos de la escena 09 y de la práctica habitual; **no están confirmados por ASICAI**. Marcar cuáles interesan, tachar las que no y añadir las que falten.

| # | Elemento (ES) | Texto para el modelo (EN) | Tipo de marca | ¿Interesa a ASICAI? |
|---|---|---|---|---|
| 1 | Poste | utility pole | caja | [ ] |
| 2 | Cruceta | crossarm | caja | [ ] |
| 3 | Aislador | insulator | caja | [ ] |
| 4 | Transformador de poste | pole-mounted transformer | caja | [ ] |
| 5 | Conductor / cable | power line cable | **máscara o línea** (una caja no sirve para un cable) | [ ] |
| 6 | Seccionador / cortacircuito | cutout fuse / disconnect switch | caja | [ ] |
| 7 | Pararrayos | surge arrester | caja | [ ] |
| 8 | Herraje / grillete / conexión | line hardware / clamp | caja | [ ] |
| 9 | Retenida (tensor) | guy wire | línea | [ ] |
| 10 | Torre (si aplica alta tensión) | transmission tower | caja | [ ] |
| 11 | Otro (indicar) | | | [ ] |

## Preguntas que definen el alcance
1. ¿"380" significa 380 V (red de distribución en baja tensión) o es otro nivel de tensión? ¿Postes de distribución o torres de transmisión?
2. ¿Qué se hace con la detección? (a) inventario de elementos, (b) localizar el elemento para medir su temperatura en la imagen térmica, (c) detectar anomalías físicas (inclinación, aislador roto…).
3. ¿Cámara visual, térmica o ambas? (Los modelos públicos de elementos de línea trabajan sobre imagen a color.)
4. ¿Cuántas fotos o videos de postes tiene ASICAI con el Mavic 3T? ¿Se pueden capturar más, y en qué zonas?
5. ¿Quién del equipo valida los nombres y las etiquetas?

## Cómo se usaría esta lista (flujo propuesto)
1. ASICAI valida las clases.
2. Un modelo con texto (OWLv2, licencia Apache 2.0) propone cajas con un umbral alto.
3. Una persona acepta o corrige con la herramienta de revisión, y se registra el tiempo por foto.
4. Con las fotos corregidas se entrena un detector propio (revisar la licencia de la librería de entrenamiento antes de un uso comercial).
5. La parte térmica se aplica sobre la zona de cada elemento encontrado: "¿se ve más caliente que su entorno?". La decisión de falla la toma un técnico.
