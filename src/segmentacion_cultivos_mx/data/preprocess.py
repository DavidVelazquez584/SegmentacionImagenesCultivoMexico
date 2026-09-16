"""Funciones iniciales para validar y preparar datos de segmentación."""

from pathlib import Path


def validate_data_directories(image_dir: Path, mask_dir: Path) -> None:
    """Verifica que existan las carpetas esperadas de imágenes y máscaras."""

    if not image_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de máscaras: {mask_dir}")


def normalize_filename(path: Path) -> str:
    """Devuelve un identificador simple basado en el nombre del archivo."""

    return path.stem.lower().replace(" ", "_")
