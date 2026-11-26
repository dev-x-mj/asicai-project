# Guía de captura propuesta — fotos térmicas y visuales (DJI Mavic 3T)

**BORRADOR.** Nace de lo que mostraron las 170 fotos actuales y de la ficha técnica del dron. Los puntos marcados con **[validar]** deben confirmarlos técnicos de ASICAI; esta guía no reemplaza sus procedimientos de vuelo, seguridad ni los permisos aplicables.

## Por qué hace falta
- Todas las fotos actuales conservan los mismos parámetros de medición (distancia 2.5 m, emisividad 0.95, temperatura ambiente y reflejada 25 °C), lo que sugiere valores por defecto.
- El telémetro no dio lectura válida en 160 de 170 fotos.
- Solo hay 16 fotos de postes, de una sola estructura, y ASICAI no tiene más.

## Qué capturar siempre
1. **Foto térmica radiométrica (R-JPEG) y foto visual a la vez.** No renombrar ni editar los archivos originales (`DJI_..._T.JPG` y `_V.JPG`).
2. **Varios lugares y varias estructuras de postes**, no solo una. Cada estructura desde varios ángulos y distancias.
3. **Cada elemento de interés completo en cuadro** (poste, cruceta, aisladores, transformador y equipos de maniobra), además de planos de detalle.

## Distancia
Según el campo de visión de la ficha técnica (aproximado), el tamaño de un píxel térmico es:
| Distancia | Píxel (cm) |
|---|---|
| 5 m | 0.72 |
| 10 m | 1.44 |
| 20 m | 2.87 |
| 30 m | 4.31 |
- Para conexiones, aisladores y elementos pequeños: **3 a 10 m** [validar con ASICAI].
- Para conductores finos (unos 2 cm, calibre real [confirmar]): del orden de **5 m** para que ocupen unos 3 píxeles; a más distancia no se miden bien.
- Las tomas generales de contexto pueden hacerse a mayor distancia; sirven para ubicar, no para medir temperatura.

## Parámetros de medición
- Ajustar la **emisividad** según el material y la **temperatura reflejada y ambiente** reales, o registrar claramente que se usaron los valores por defecto [validar con técnico].
- Verificar que el **telémetro apunte al objetivo** y devuelva lectura; si no, anotar la distancia estimada.
- Evitar **cielo en el cuadro** en las tomas térmicas: un cielo muy frío distorsiona la escala (hubo fotos con mínimos de −12 a −19 °C) y puede afectar a los algoritmos.
- Evitar **reflejos del sol** sobre superficies metálicas o brillantes.
- La hora y el sol afectan lo que se ve: anotar las condiciones; si conviene evitar horas de mucho sol, decidirlo con el técnico [validar].

## Registro por sesión (un archivo CSV o una tabla)
| Campo | Ejemplo |
|---|---|
| Fecha y hora de inicio/fin | [ ] |
| Lugar / identificador de la estructura | [ ] |
| Tipo de elemento | poste, tablero, transformador, otro |
| Clima (temperatura, viento, nubosidad, sol) | [ ] |
| Carga o estado de operación del equipo, si se conoce | [ ] |
| Emisividad y temperatura reflejada usadas | [ ] |
| Distancia aproximada | [ ] |
| Piloto / responsable | [ ] |
| Notas | [ ] |

## Organización y confidencialidad
- Una carpeta por sesión; conservar los nombres originales del dron.
- Guardar las fotos fuera del repositorio de código.
- Revisar el reloj del dron antes de volar (la carpeta `FOTOS` tiene fechas que pueden ser incorrectas).

## Cantidad
Cuántas fotos hacen falta depende del desempeño que se quiera y no se puede fijar hoy. Como referencia propia: etiquetar unas 8 fotos de postes tomó unos 6 minutos por foto, con unas 11 cajas por foto (una sola sesión, aproximado). Una primera meta razonable debe definirla ASICAI **[por definir]**.
