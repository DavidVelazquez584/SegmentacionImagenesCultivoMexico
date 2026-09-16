# Segmentación Semántica de Cultivos en Imágenes Satelitales

Repositorio base para el proyecto:

```text
Segmentación semántica de cultivos en imágenes satelitales y generación automática de reportes mediante Deep Learning.
```

## Descripción

El proyecto busca analizar imágenes satelitales o series temporales de parcelas agrícolas, segmentar o clasificar cultivos y generar reportes textuales a partir de los resultados obtenidos.

## Datos generales

- Institución: Tecnológico de Monterrey.
- Programa: Maestría en Inteligencia Artificial Aplicada.
- Dominio principal: visión computacional.
- Sector de aplicación: agricultura de precisión.
- Alcance inicial: México.
- Sponsor académico: Dr. Gerardo Jesús Camacho González.
- Colaboración académica: Scuola Superiore Sant'Anna, Pisa, Italia.

## Objetivo

Desarrollar una base de trabajo para procesar imágenes satelitales de cultivos, ejecutar modelos de segmentación o clasificación, calcular métricas por cultivo o parcela y generar reportes en lenguaje natural.

## Flujo propuesto

```text
imagenes satelitales o series temporales
    -> preparacion de datos
    -> modelo de segmentacion o clasificacion
    -> predicciones por pixel, parcela o cultivo
    -> calculo de area, cobertura y confianza
    -> reporte textual
```

## Referencias base

- ViTs for SITS: Vision Transformers for Satellite Image Time Series  
  <https://arxiv.org/abs/2301.04944>

- DeepSatModels  
  <https://github.com/michaeltrs/DeepSatModels>

- "Phenology description is all you need!" mapping unknown crop types with remote sensing time-series and LLM generated text alignment  
  <https://www.sciencedirect.com/science/article/abs/pii/S0924271625002643>

## Estructura

```text
.
├── configs/                         # Configuraciones del proyecto
├── data/                            # Datos locales
│   ├── external/
│   ├── interim/
│   ├── masks/
│   ├── processed/
│   └── raw/
├── docs/                            # Documentación y referencias
│   ├── onboarding/
│   ├── research/
│   └── schemas/
├── models/                          # Modelos y checkpoints
│   └── checkpoints/
├── notebooks/                       # Exploración y pruebas
├── reports/                         # Métricas, figuras y resultados
│   ├── figures/
│   └── metrics/
├── scripts/                         # Scripts ejecutables
├── src/segmentacion_cultivos_mx/    # Código fuente principal
├── tests/                           # Pruebas básicas
└── third_party/                     # Repositorios externos
```

## Uso inicial

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Verificación básica:

```bash
python -m compileall scripts src tests
PYTHONPATH=src python scripts/generate_text_report.py
```

Repositorio externo base:

```bash
git clone https://github.com/michaeltrs/DeepSatModels.git third_party/DeepSatModels
```
