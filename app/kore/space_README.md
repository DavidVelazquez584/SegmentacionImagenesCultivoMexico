---
title: Koré
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Conversa con mapas de cultivos segmentados por TSViT (PASTIS)
---

# Koré — cultivos desde el espacio

Koré segmenta **en vivo** un parche de PASTIS (serie de tiempo Sentinel-2 de 128 × 128 píxeles, Francia
2018–2019) con un modelo **TSViT** entrenado desde cero por el Equipo 13 (Tec de Monterrey), y permite
conversar sobre el resultado: hectáreas por cultivo, cultivo dominante, confianza del modelo y comparación
con la etiqueta real.

- **Modelo de segmentación:** TSViT (Tarasiou et al., CVPR 2023), época 57. Conjunto de prueba (fold 5):
  mIoU 0.6587, Macro F1 0.7807, Accuracy 0.8378.
- **Chat:** Claude (Anthropic) recibe la tabla calculada en Python y solo explica esos números.
- **Regla anti-fuga:** solo los parches del fold 5 (nunca vistos al entrenar) se comparan con su etiqueta
  real; para los demás no se muestran métricas.
- **Limitaciones:** el modelo solo es válido para datos tipo PASTIS. Las hectáreas son estimaciones del
  modelo (1 píxel = 0.01 ha), no mediciones de campo. Las clases forraje leguminoso, frutas/hortalizas,
  sorgo y cereal mixto tienen IoU bajo (0.37–0.47).

## Créditos y licencias
- **Datos:** PASTIS — V. Sainte Fare Garnot y L. Landrieu, *Panoptic Segmentation of Satellite Image Time
  Series with Convolutional Temporal Attention Networks*, ICCV 2021. Licencia
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Este Space redistribuye un subconjunto
  (8 series de ejemplo, metadatos y etiquetas del fold 5). Imágenes: datos Copernicus Sentinel-2 procesados
  por CNES para el polo de datos Theia; parcelas: Registro Parcelario Gráfico (IGN).
- **Código de TSViT:** github.com/michaeltrs/DeepSatModels, licencia Apache 2.0.
