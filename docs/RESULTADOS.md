# Resultados de la validación (sprint actual)

## Métricas obtenidas

Sobre un set de 8 escenas térmicas sintéticas (17 hotspots conocidos en total):

| Métrica | Valor |
|---|---|
| Verdaderos positivos | 17 |
| Falsos negativos | 0 |
| Falsos positivos | 0 |
| Precisión | 1.00 |
| Recall | 1.00 |
| F1-score | 1.00 |

## Cómo interpretar esto correctamente (importante)

Estas métricas **no significan que el sistema esté listo para
producción**. Significan que:

- El pipeline completo funciona de punta a punta sin errores.
- La lógica de detección, clasificación de severidad y evaluación es
  correcta sobre un caso controlado donde conocemos la respuesta exacta.

Es decir: **validamos la arquitectura e implementación, no el
desempeño en el mundo real.** Las imágenes sintéticas son más simples y
limpias que una imagen térmica real de un dron (que tendrá ruido,
reflejos, múltiples materiales, condiciones de viento/distancia
variables, etc.).

## Qué significa esto para el siguiente sprint

El resultado de 1.00/1.00 es exactamente la señal esperada para "el
sistema base está listo" — el siguiente paso correcto es poner a prueba
el mismo pipeline con **imágenes reales de ASICAI**, donde es normal y
esperable que la precisión/recall bajen inicialmente. Ahí es donde el
ajuste de umbrales (o el salto a un modelo YOLO entrenado) aporta valor
real.
