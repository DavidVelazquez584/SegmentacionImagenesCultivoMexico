"""Pruebas iniciales para validar la estructura del repositorio."""

from pathlib import Path


def test_expected_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_directories = [
        "configs",
        "data/raw",
        "data/masks",
        "data/processed",
        "docs/onboarding",
        "docs/research",
        "docs/schemas",
        "models/checkpoints",
        "notebooks",
        "reports/figures",
        "reports/metrics",
        "scripts",
        "src/segmentacion_cultivos_mx",
        "third_party",
    ]

    for directory in expected_directories:
        assert (root / directory).is_dir()
