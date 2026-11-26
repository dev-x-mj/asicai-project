# Plan de cierre — ASICAI PoC termografía (Mavic 3T)

Leyenda: **[C]** comprobado · **[H]** hipótesis (hay que verificarla) · **[?]** no lo sé / lo completas tú.

---
## 1. Dónde estás realmente

| Tema | Estado |
|---|---|
| Dron / cámara | **[C]** Mavic 3 Enterprise Thermal (3T) según datasheet: térmica 640x512, píxel 12 µm, DFOV 61°, precisión ±2 °C o ±2 %, R-JPEG 16 bits, rango −20 a 150 °C (alta ganancia) / 0 a 500 °C (baja). |
| PoC con datos sintéticos | **[C]** Funciona de punta a punta (informe de avance + notebook). |
| Desempeño en fotos reales | **[?]** Aún no medido. Es lo que falta y lo que más vale. |
| Fotos reales autorizadas | **[C]** Existen (`fotos_publicas.7z`). **[?]** Cuántas y de cuántas escenas: lo dirá el inventario. |
| Temperatura real por píxel | **[H]** Se puede sacar de los `_T.JPG` (R-JPEG) con el DJI Thermal SDK. No lo he podido probar aquí. |

## 2. Correcciones a lo anterior (importante que las sepas antes de defender)

1. **Notebook con resultados inconsistentes.** Configuración base: recall 0.95 (FN=3). Pero el Experimento B con `min_area_px=20` (los mismos parámetros) muestra recall 0.933 (FN=4). Con semillas fijas eso no puede pasar: **[H]** las salidas son de ejecuciones distintas del código. Solución: *Reiniciar y ejecutar todo* y guardar de nuevo.
2. **"F1 = 1.00" del informe de avance no es general.** Salió de 8 escenas / 17 puntos. En 30 escenas el mismo detector da recall 0.95. Cita ambos y di que el simulador y el detector se diseñaron juntos (resultado optimista por construcción).
3. **"Severidad alta/media/baja"**: es una heurística por intensidad, no una severidad de falla (depende de carga, emisividad, ΔT contra referencia, normativa). Renómbrala **"prioridad de revisión sugerida"**. Decidir si algo es falla le toca a un técnico de ASICAI.
4. **Experimento B**: el área mínima no afecta hasta 300 px porque los puntos sintéticos son grandes. En el dron el tamaño depende de la distancia (tabla en §6 del informe de viabilidad). El experimento no es representativo del campo.
5. **Informe de avance desactualizado**: ya hay modelo de dron confirmado y fotos reales autorizadas. Actualiza esas "dificultades". Verifica también el PyTorch (tu venv es Python 3.13.7) **[?]**.
6. **Lo que se ve en tus miniaturas [H]**: las dos sesiones usan paletas distintas (una tipo hierro/fuego, otra tipo arcoíris). Convertir a gris una paleta arcoíris no conserva el orden frío→caliente. Por eso el modo "proxy" (gris) del código es solo para probar el flujo; para resultados reales hay que usar °C del SDK. Además la cámara visual (`_V`) tiene otro campo de visión que la térmica: no se superponen directo.
7. **El detector por percentil siempre "encuentra algo"** en la parte más caliente de cualquier foto. Mide cuántas fotos *sanas* dan detecciones (columna `n_detecciones` de `detections.csv`): es tu estimación de falsos positivos.

## 3. ¿Entrenar con el resto de fotos? Respuesta corta

- **Usar TODAS las fotos: sí**, para inventario y estadísticas del baseline (sin etiquetas).
- **Medir desempeño real: sí, obligatorio.** Necesitas fotos *revisadas a mano* (ground truth). Sin eso no hay métrica real.
- **Entrenar YOLO: opcional, y solo sobre fotos revisadas.** Entrenar con las pre-etiquetas del baseline sin revisar solo enseña al modelo los errores del baseline. Con pocas escenas el resultado será **no concluyente**, y eso es un hallazgo legítimo para el informe de viabilidad (qué volumen de datos hace falta). Regla práctica mía **[H]**: con menos de ~30 fotos revisadas o ~8 escenas distintas, hazlo como "ejercicio de factibilidad" y repórtalo así.
- **Nunca** subas fotos a Colab/Roboflow/ningún servicio en la nube sin permiso escrito de ASICAI.

## 4. Orden de trabajo (estimaciones de horas = mías, ajusta con tu experiencia real)

**Bloque 0 — Preparación (~1 h)**
1. Copia `src/*.py` de este paquete a `asicai-poc/src/` y `tests/` a `asicai-poc/tests/`.
2. Extrae el `.7z` con 7-Zip **fuera del repo** (ej. `C:\ASICAI_privado\fotos_publicas`).
3. Añade a `.gitignore`: `data/real_private/` y `*.JPG`. Nunca se hace commit de fotos reales.
4. `pip install -r requirements_extra.txt` y `python tests/test_pipeline.py` (debe terminar en `TODO OK`).

**Bloque 1 — Inventario + baseline en modo proxy (~1–2 h)**
```
python src/real_baseline.py --root "C:\ASICAI_privado\fotos_publicas" --out data\real_private
python src/gsd.py
```
Mira `data\real_private\overlays\`. Anota: n.º de fotos, n.º de escenas, fotos sin detecciones.

**Bloque 2 — Temperatura real con el SDK (~2–4 h, es el paso más incierto)**
- Descarga el DJI Thermal SDK del portal de desarrolladores de DJI **[?] verifica licencia y versión**. Lee su README y prueba `dji_irp --help`: mi llamada en `real_data._load_with_sdk` está escrita de memoria y puede necesitar ajuste.
- Corre de nuevo con `--sdk "C:\ruta\dji_irp.exe"`. Si hay datos de distancia/emisividad de cada sesión [?], pásalos con `sdk_extra_args`.
- Si no logras el SDK en un tiempo razonable: documenta el bloqueo, sigue en modo proxy y declara en el informe que los resultados son **sobre brillo, no sobre °C** (limitación, no fracaso).

**Bloque 3 — Ground truth (~3–5 h)**
```
python src/review_labels.py --out data\real_private --blind      # primeras ~15 fotos: sin sesgo
python src/review_labels.py --out data\real_private              # el resto: corrige las cajas del baseline
```
Criterio de etiqueta: "zona más caliente que su entorno que un técnico querría revisar". Pide a un técnico de ASICAI que valide ese criterio y, si puede, una muestra de tus etiquetas **[?]**. Ojo: si corriges las cajas del baseline, sus métricas salen algo infladas (sesgo de anclaje); por eso el modo `--blind`. Reporta ambas cosas.

**Bloque 4 — Métricas reales del baseline (~1–2 h)**
```
python src/make_yolo_dataset.py --out data\real_private
python src/eval_real.py --out data\real_private --pred-dir data\real_private\labels_pre --stems data\real_private\val_stems.txt
```
Haz también un barrido de `std_k`/`percentile`/`min_area` sobre fotos reales (como tus Experimentos A y B) y compáralo con los sintéticos. Ese contraste es contenido de oro para el informe.

**Bloque 5 — YOLO (opcional, ~3–6 h)**
```
pip install ultralytics
python src/train_yolo.py --out data\real_private --epochs 100 --device cpu
python src/train_yolo.py --out data\real_private --predict runs\asicai\hotspot\weights\best.pt
python src/eval_real.py --out data\real_private --pred-dir data\real_private\pred_yolo --stems data\real_private\val_stems.txt
```
(Probablemente necesites ajustar si ultralytics/torch no instalan en Python 3.13; en ese caso usa un venv con Python 3.11/3.12.) Compara baseline vs YOLO **en las mismas fotos de validación**. Si YOLO no gana o la validación es diminuta, dilo tal cual.

**Bloque 6 — Cierre (~6–10 h)**
- Re-ejecutar el notebook completo y completar tus interpretaciones con tus palabras (cap. 6, 7 y 9).
- Ejecutar `resultados_reales.ipynb` (raíz del proyecto, junto a `experimentos.ipynb`, que se conserva para los sintéticos). Rellenar `BLIND_N` y tus interpretaciones. Antes de cualquier commit: `pip install nbstripout` y `nbstripout resultados_reales.ipynb`. Las tablas numéricas quedan en `data/real_private/tablas/` para el informe.
- Actualizar `README.md`, `ARCHITECTURE.md`, `RESULTADOS.md`, `NEXT_STEPS.md`.
- Escribir el **informe de viabilidad** (esqueleto en `INFORME_VIABILIDAD_esqueleto.md`).
- Informe de avance final + firma del tutor + registro de horas.

**Segundo caso de uso (líneas eléctricas, etc.):** no lo implementes. Descríbelo en `NEXT_STEPS.md` con lo aprendido (qué datos, qué etiquetas, qué riesgos). Tus fotos del tablero/transformador pueden servir de ejemplo ilustrativo.

## 5. Checklist para dar por terminadas las prácticas
- [ ] Repo limpio, sin fotos reales, con `README` que permita reproducir.
- [ ] Tests pasan (`python tests/test_pipeline.py`).
- [ ] Notebook re-ejecutado y con tus conclusiones.
- [ ] Métricas reales del baseline (aunque sean sobre pocas fotos) con sus límites explicados.
- [ ] Informe de viabilidad completo (sin datos inventados; huecos marcados).
- [ ] Informe de avance final firmado por el tutor.
- [ ] Registro de horas honesto (§7).
- [ ] Confirmación escrita de ASICAI sobre el uso de las fotos.

## 6. Preguntas para tu tutor (mándalas ya, la respuesta puede tardar)
1. ¿Autorizan usar las fotos del `.7z` para evaluar y para entrenar un modelo? ¿Puede ese modelo/derivados publicarse en la memoria de UNIR (solo figuras anonimizadas)?
2. ¿Quién del equipo puede validar mi criterio de "punto de interés" y revisar una muestra de etiquetas?
3. Por sesión: distancia al equipo, emisividad/ajustes usados, hora, carga del equipo, paleta [?]. Sin esto, los °C absolutos no son interpretables.
4. ¿Qué entregable mínimo da por cumplido el reto (PoC + informe + docs)?

## 7. Registro de horas (honesto)
Anota el tiempo **real** al terminar cada sesión; si dudas, redondea hacia abajo. Aquí cuenta lo que hiciste tú (leer, probar, depurar, escribir), no el tiempo que tardó un proceso en correr mientras hacías otra cosa.

| Fecha | Inicio | Fin | Horas | Actividad | Evidencia (commit / archivo) |
|---|---|---|---|---|---|
| [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Total acumulado: [ ] / 96 h
