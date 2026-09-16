"""Funciones para visualizar imágenes, máscaras y predicciones."""

from pathlib import Path


def describe_visual_comparison(image_path: Path, mask_path: Path, prediction_path: Path | None = None) -> str:
    """Describe una comparación visual esperada."""

    parts = [f"imagen={image_path}", f"máscara={mask_path}"]
    if prediction_path is not None:
        parts.append(f"predicción={prediction_path}")
    return ", ".join(parts)
