"""Genera un reporte textual preliminar con datos simulados de fase 1."""

from segmentacion_cultivos_mx.analysis.parcel_summary import CropSummary
from segmentacion_cultivos_mx.reporting.narrative import generate_crop_report


def main() -> None:
    crops = [
        CropSummary("maiz", 18.4, 63.9, 0.86, 12),
        CropSummary("trigo", 7.2, 25.0, 0.79, 5),
        CropSummary("sin cultivo aparente", 3.2, 11.1, 0.71, 3),
    ]
    notes = [
        "Ejemplo simulado para validar el formato del reporte.",
        "En fases posteriores estos valores deben venir de mascaras o predicciones reales.",
    ]
    print(generate_crop_report("zona agricola de ejemplo en Mexico", crops, notes))


if __name__ == "__main__":
    main()
