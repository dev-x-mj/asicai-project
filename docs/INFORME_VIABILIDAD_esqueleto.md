# Informe técnico de viabilidad — IA en drones termográficos (ASICAI)

**Autor:** Christian Masaquiza Jerez · **Tutor:** Ing. Andrés Álvarez (ASICAI) · **Fecha:** [ ]
Convenciones: **[C]** comprobado con dato · **[H]** hipótesis · **[?]** completar. Todo lo que dependa de fotos reales se rellena con TUS resultados; no pongas cifras que no hayas medido.

## 1. Resumen ejecutivo
[Pregunta del reto · qué se construyó · qué se midió en fotos reales (n fotos, n escenas) · conclusión de viabilidad en 3–4 líneas · qué falta]

## 2. Contexto y alcance
- Empresa y reto: ASICAI, mantenimiento predictivo e inspección industrial con drones termográficos.
- Alcance: detección de zonas calientes para **apoyar** la revisión de un técnico. **No** diagnostica fallas.

## 3. Equipo de captura (datasheet DJI Mavic 3T) [C]
| Característica | Valor |
|---|---|
| Sensor térmico | Microbolómetro VOx no refrigerado, 640×512, píxel 12 µm, 30 Hz |
| Campo de visión (DFOV) | 61° |
| Sensibilidad (NETD) | ≤ 50 mK @ F1.1 |
| Rango de medición | −20 a 150 °C (alta ganancia) · 0 a 500 °C (baja ganancia) |
| Precisión | ±2 °C o ±2 % (el mayor) |
| Formato de foto | JPEG 8 bits y **R-JPEG 16 bits** (radiométrico) |
| Zoom digital térmico | 28× |
| Tiempo de vuelo | hasta 45 min (sin viento) |
| Operación | −10 a 40 °C; resistencia al viento 12 m/s |

Implicación: la resolución térmica es baja (0.33 MP) frente a la cámara visual de 48 MP. Cualquier detector tiene un límite físico de lo que "ve".

## 4. Datos
### 4.1 Sintéticos [C]
8 escenas/17 puntos: P=R=F1=1.00. 30 escenas: P=1.00, R=0.95 (3 FN). Limitación: el simulador y el detector se diseñaron juntos.
### 4.2 Reales (muestra autorizada) [?]
- N.º de fotos térmicas: [ ] · escenas: [ ] · sesiones/fechas: [ ] · tipos de equipo: [ ]
- Dato usado: [°C vía SDK | brillo (proxy)] · Paletas encontradas: [ ]
- Datos de captura disponibles (distancia, emisividad, carga): [ ]
- Ground truth: [n fotos revisadas, n cajas, quién validó el criterio, % hecho a ciegas]

## 5. Método
Pipeline: ingesta → lectura térmica → detección → priorización → reporte.
Baseline clásico: umbral = max(percentil, media + k·σ) + componentes conectados + área mínima. YOLO [si se hizo]: modelo [ ], épocas [ ], dataset [ ], partición por escena.

## 6. Efecto de la altura de vuelo (tamaño del punto en píxeles) [C: cálculo a partir del datasheet; H: aproximación]
IFOV derivado de DFOV=61° y 640×512: ≈ 1.44 mrad/píxel (el datasheet no da la focal real).
| Distancia (m) | Píxel (cm) | Campo horizontal (m) | Objeto mín. 3 px (cm) | Diámetro de un punto de 20 px (cm) |
|---|---|---|---|---|
| 5 | 0.72 | 4.6 | 2.2 | 3.6 |
| 10 | 1.44 | 9.2 | 4.3 | 7.3 |
| 20 | 2.87 | 18.4 | 8.6 | 14.5 |
| 30 | 4.31 | 27.6 | 12.9 | 21.8 |
| 50 | 7.19 | 46.0 | 21.6 | 36.3 |
Lectura: a 30 m, `min_area_px=20` ya exige una zona caliente de ~22 cm de diámetro. Para tornillería, bornes o conexiones pequeñas hay que volar cerca o usar zoom. Para medir temperatura con fiabilidad, el objeto debe cubrir varios píxeles (regla práctica común en termografía [H], verificar con el manual del fabricante).

## 7. Resultados
### 7.1 Baseline en fotos reales [?]
| Conjunto | Fotos | Cajas reales | TP | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Validación (IoU≥0.3) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
Fotos "sanas" con detecciones (falsos positivos por foto): [ ]
Contraste con sintéticos: [qué cambió y por qué]
### 7.2 YOLO vs baseline (mismas fotos de validación) [?]
[tabla + figura + comentario honesto sobre tamaño de la muestra]
### 7.3 Errores típicos observados [?]
[reflejos, objetos calientes por diseño (motores), bordes, paleta, distancia… con 2–3 ejemplos autorizados]

## 8. Limitaciones [completar con lo que pase]
- Muestra pequeña y de pocas escenas; fotos casi duplicadas dentro de una escena.
- Sin diagnóstico: un punto caliente no es una falla; depende de carga, emisividad, referencia y criterio del técnico.
- Temperatura absoluta sujeta a emisividad, reflejos, distancia y condiciones ambientales [H general].
- Etiquetas hechas por [una persona]; criterio [validado/no validado] por ASICAI.

## 9. Viabilidad: conclusión
[Técnica · Datos · Operativa · Económica (no calcular costos sin datos de ASICAI)]. Responde: ¿qué sí se puede hoy? ¿qué falta para producción? Cada afirmación apoyada en una tabla/figura de arriba.

## 10. Recomendaciones y próximos pasos
1. Protocolo de captura estandarizado (distancia, ángulo, emisividad, hora, carga, paleta única o R-JPEG siempre).
2. Registrar metadatos por sesión.
3. Plan de etiquetado con validación técnica; objetivo de volumen de datos [ ].
4. Segundo caso de uso: [líneas eléctricas / otro] con lo aprendido.
5. Integración: flujo de reporte para el técnico, revisión humana obligatoria.

## 11. Anexos
Repositorio y cómo reproducir · Datasheet · Registro de horas.
