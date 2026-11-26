# Guía de etiquetado de zonas calientes (fotos térmicas)

**Qué se marca:** una zona que se ve **claramente más caliente que su entorno inmediato en la misma foto** y que un técnico querría revisar. **No es un diagnóstico de falla.**
Las reglas numéricas de abajo son un **criterio de trabajo del autor, a validar por un técnico de ASICAI**; no son una norma.

## Comando
```
python src\review_labels.py --out data\real_private --blind --per-group 1
```
Elige 1 foto por escena (14 en total). `--blind` significa que la ventana **no** muestra las cajas del detector: marcas tú solo.

## Controles
- Arrastrar con clic izquierdo: dibuja una caja. Clic derecho sobre una caja: la borra.
- `d` o espacio: guarda y pasa a la siguiente. `a`: vuelve. `c`: borra todas las cajas de la foto. `q`: guarda y sale.
- Una foto sin cajas es válida (cuenta como "no hay zona que destaque").

## Cómo leer la imagen
- Paleta única (inferno): negro y morado = más frío; naranja, amarillo y blanco = más caliente. **Es relativa a cada foto:** el mismo color no significa la misma temperatura en dos fotos distintas.
- **Pasa el mouse sobre la imagen:** la barra superior muestra los °C bajo el cursor y el mínimo y máximo de la foto. Mide la zona sospechosa y compárala con lo que la rodea.
- Los °C absolutos son orientativos (todas las fotos conservan emisividad 0.95, distancia 2.5 m y 25 °C de ambiente por defecto). Lo defendible es **comparar una zona con su entorno**.

## Cuándo marcar
1. La zona destaca claramente respecto de lo que la rodea (diferencia visible en color **y** en °C al pasar el mouse). Como regla de trabajo inicial: unos 5 °C o más por encima del entorno inmediato [criterio del autor; validar con técnico].
2. La caja encierra solo la zona caliente, no todo el equipo.
3. Si hay varias zonas separadas, una caja por zona.

## Cuándo NO marcar
- **Todo el equipo está caliente de forma pareja** (un motor o transformador en operación): no hay zona que destaque. Foto sin caja y a la lista de dudosas.
- **Reflejos:** superficies metálicas brillantes pueden mostrar "calor" que en realidad es reflejo de otra fuente o del sol.
- **Fondo calentado por el sol**, cielo, vegetación o paredes.
- **Bordes y objetos pequeños** de pocos píxeles: poco fiables.
- Si **no estás seguro**, no marques y anótalo.

## Lista de dudosas (para el técnico)
Anota: nombre de la foto, qué se ve y por qué dudas. Por ejemplo: `20250528111651_0037 — equipo cilíndrico caliente en conjunto; ¿es normal?`.

## Qué no hacer
- No decidas si algo es una falla ni qué tan grave es.
- No subas las fotos a otros servicios ni chats.

## Al terminar
```
python src\eval_real.py --out data\real_private --pred-dir data\real_private\labels_pre
```
Imprime precisión, recall y F1 del detector baseline sobre todas las fotos revisadas, con tres umbrales de IoU. Los parámetros del detector no se ajustan sobre estas fotos, así que no hay sobreajuste; aun así, son pocas fotos y el resultado es inestable.
