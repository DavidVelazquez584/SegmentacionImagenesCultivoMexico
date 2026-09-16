"""Generación inicial de reportes narrativos agrícolas."""

from segmentacion_cultivos_mx.analysis.parcel_summary import (
    CropSummary,
    get_dominant_crop,
    total_area_hectares,
)


def generate_crop_report(zone_name: str, crops: list[CropSummary], notes: list[str] | None = None) -> str:
    """Genera un reporte textual simple a partir de estadísticas por cultivo."""

    if not crops:
        return f"No hay cultivos detectados para la zona {zone_name}."

    total_area = total_area_hectares(crops)
    dominant_crop = get_dominant_crop(crops)
    crop_lines = [
        (
            f"- {crop.crop_name}: {crop.area_hectares:.2f} ha "
            f"({crop.coverage_percent:.1f}% de cobertura), "
            f"{crop.detected_parcels} parcelas detectadas, "
            f"confianza promedio {crop.mean_confidence:.2f}."
        )
        for crop in sorted(crops, key=lambda item: item.area_hectares, reverse=True)
    ]

    narrative = [
        f"Reporte preliminar para {zone_name}.",
        (
            f"La superficie total analizada es de aproximadamente {total_area:.2f} ha. "
            f"El cultivo o clase dominante es {dominant_crop.crop_name}, con "
            f"{dominant_crop.area_hectares:.2f} ha estimadas."
        ),
        "Distribución estimada por clase:",
        *crop_lines,
        (
            "La interpretación debe considerarse preliminar porque depende de la calidad "
            "de las imágenes, fechas disponibles, bandas satelitales y validación de campo."
        ),
    ]

    if notes:
        narrative.append("Observaciones:")
        narrative.extend(f"- {note}" for note in notes)

    return "\n".join(narrative)
