"""Utilidades para resolver rutas del repositorio."""

from pathlib import Path


def get_project_root() -> Path:
    """Devuelve la ruta raíz del repositorio."""

    return Path(__file__).resolve().parents[3]


def ensure_directory(path: Path) -> Path:
    """Crea una carpeta si no existe y devuelve la misma ruta."""

    path.mkdir(parents=True, exist_ok=True)
    return path
