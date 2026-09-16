"""Estructuras para resumir predicciones por cultivo o parcela."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CropSummary:
    """Resumen agregado de una clase de cultivo detectada."""

    crop_name: str
    area_hectares: float
    coverage_percent: float
    mean_confidence: float
    detected_parcels: int


def get_dominant_crop(crops: list[CropSummary]) -> CropSummary | None:
    """Devuelve el cultivo con mayor superficie estimada."""

    if not crops:
        return None
    return max(crops, key=lambda crop: crop.area_hectares)


def total_area_hectares(crops: list[CropSummary]) -> float:
    """Suma la superficie estimada para todos los cultivos reportados."""

    return sum(crop.area_hectares for crop in crops)
