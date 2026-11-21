# Koré: mapas de cultivos y chat sobre imágenes Sentinel-2

Koré ejecuta en vivo el **TSViT entrenado desde cero** (Avance 7) sobre un parche de PASTIS:
- muestra la imagen en color verdadero, el mapa de cultivos predicho y las hectáreas por clase;
- permite conversar con un modelo de lenguaje acotado a esas cifras.

El nombre viene de Koré (Perséfone), cuyo regreso cada primavera marca el ciclo de los cultivos.

> **Alcance.** El modelo se entrenó con cultivos de Francia (PASTIS, temporada 2018–2019). Las hectáreas son estimaciones del modelo, no mediciones de campo, y las métricas no se extrapolan a México sin una validación local.

## Resultados del modelo (fold 5, 496 parches nunca vistos al entrenar)

| Modelo | mIoU | Macro F1 | Accuracy |
|---|---|---|---|
| **TSViT desde cero (el de Koré)** | **0.6587** | **0.7807** | **0.8378** |
| TSViT, pesos oficiales | 0.6509 | 0.7745 | 0.8337 |
| U-TAE (Avance 6) | 0.6424 | 0.7703 | 0.8290 |

- **Clases débiles** (IoU < 0.50): frutas/hortalizas, forraje leguminoso, cereal mixto, huerto y sorgo. Koré lo advierte en el chat.
- **Equivalencia con el notebook:** la inferencia de [`inferencia.py`](inferencia.py) reproduce el notebook del Avance 7 en los 496 parches, con matrices de confusión idénticas (`pruebas/equivalencia_fold5_gpu.log`).

## Estructura

| Archivo | Qué hace |
|---|---|
| `app.py` | Interfaz Gradio: galería, subida de archivos, mapas, tabla, indicadores y chat |
| `inferencia.py` | Carga el checkpoint, ensambla 36 ventanas de 24×24 (con TTA opcional) y resume hectáreas por clase |
| `contexto.py` | Arma el *prompt* del asistente con la tabla del parche y las reglas para no inventar cifras |
| `llm.py` | Motores de lenguaje: Ollama (local), Anthropic (API) o un modo de prueba sin LLM |
| `estilo.py` | Tema, paleta, CSS, botón de modo claro/oscuro y pantalla de inicio de sesión |
| `modelo/` | `tsvit_modelo.py` (arquitectura) y `tsvit_best.pt` (checkpoint propio, época 57, 6.7 MB) |
| `datos/` | `metadatos.json` (fold y fechas de los 2,433 parches) y `etiquetas_fold5.npz` |
| `preparar_datos.py` | Regenera `datos/` desde PASTIS, incluidos los 8 ejemplos de la galería |
| `armar_space.py` | Arma la carpeta para Hugging Face Spaces (`space_README.md`, `space_requirements.txt`) |
| `pruebas/` | Pruebas de la app, del LLM, de equivalencia y de tiempos, con sus registros |

## Instalación

Requisitos:
- Python 3.12;
- PyTorch (con CUDA si hay GPU);
- `gradio==6.32.0`, `numpy`, `pandas`, `einops`, `anthropic`.

Para CPU basta con `space_requirements.txt` más `gradio==6.32.0`:

```bash
pip install -r space_requirements.txt gradio==6.32.0
```

Los parches de ejemplo (unos 120 MB) **no están en el repositorio**. Para generarlos hay que tener PASTIS descargado ([Zenodo](https://doi.org/10.5281/zenodo.5012942)). Indica su carpeta y genera los datos:

```bash
set PASTIS_ROOT=C:\ruta\a\PASTIS
python preparar_datos.py
```

## Uso (guía operativa)

1. **Elegir el motor de lenguaje** con la variable `LLM_BACKEND`:

   | Valor | Motor |
   |---|---|
   | `auto` (por defecto) | Ollama si está corriendo; si no, la API de Anthropic si hay clave; si no, el modo de prueba |
   | `ollama` | Modelo local `ministral-3:3b` (cambiar con `OLLAMA_MODELO`). Instalar Ollama y ejecutar `ollama pull ministral-3:3b` |
   | `api` | Claude por la API de Anthropic (`ANTHROPIC_MODELO`, `ANTHROPIC_ESFUERZO`) |
   | `prueba` | Sin modelo de lenguaje, solo para revisar la interfaz |

   Para `api`, la clave va **solo** en la variable de entorno `ANTHROPIC_API_KEY` (o en un *secret* del Space), nunca en el código ni en el repositorio.
2. **Ejecutar** `python app.py` y abrir http://127.0.0.1:7860.
3. **Analizar:** elegir un ejemplo de la galería, o subir un archivo `S2_<ID>.npy` de PASTIS, y pulsar **Analizar parche**. El nombre del archivo debe conservar el ID, porque con él se recuperan las fechas de adquisición.
4. **Revisar** los mapas, la tabla de hectáreas y los indicadores, y preguntar en el chat; por ejemplo, *¿cuántas hectáreas son de maíz?*
5. **Compartir temporalmente** (opcional), con un enlace `*.gradio.live` que dura hasta una semana mientras la computadora siga encendida:

   ```bash
   set KORE_COMPARTIR=1
   set KORE_USUARIO=equipo13
   set KORE_CLAVE=<elige-una-clave>
   python app.py
   ```

   No guardes la clave en el repositorio.

### Regla contra la fuga de datos

Solo los parches del **fold 5** (prueba) se comparan con su etiqueta real. Para los folds 1–4, que se usaron para entrenar o elegir el checkpoint, Koré muestra la predicción pero **oculta la etiqueta real y las métricas**. La prueba `pruebas/prueba_app.py` lo verifica.

### Reglas del asistente

El asistente:
- responde solo con las cifras de la tabla del parche;
- advierte cuando una clase es débil o la confianza es menor a 0.80;
- no da precios, rendimientos, costos ni clima;
- no inventa causas.

## Pruebas

```bash
python pruebas/prueba_app.py              # anti-fuga, validaciones, chat sin parche, fotos sin nubes
python pruebas/prueba_llm.py              # 11 preguntas con verificación automática (usa el LLM configurado)
python pruebas/prueba_equivalencia.py     # 496 parches del fold 5 contra el notebook (requiere PASTIS)
python pruebas/prueba_tiempos_resumen.py  # tiempos por parche en CPU/GPU
```

Resultados registrados:
- Ministral 3 3B (Ollama): 11/11 verificaciones.
- Tiempo con GPU: 1.73 s por parche con TTA (RTX 4050 portátil).
- Tiempo con CPU: unos 3 s por parche sin TTA.

## Créditos y licencias

- **Datos:** PASTIS (Sainte Fare Garnot y Landrieu, ICCV 2021), licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Imágenes Copernicus Sentinel-2 procesadas por CNES/Theia; parcelas del RPG (IGN).
- **Modelo:** TSViT (Tarasiou et al., CVPR 2023), código original bajo Apache 2.0. El checkpoint de esta carpeta lo entrenó el Equipo 13 desde cero; **no** incluye los pesos oficiales de los autores.
- **Desarrollo:** Equipo 13, Maestría en Inteligencia Artificial Aplicada, Tecnológico de Monterrey. Con apoyo de Claude Code (Claude Opus 5.5, Anthropic) en la programación.
