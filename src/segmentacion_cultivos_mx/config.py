"""Configuración base y rutas principales del proyecto."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Rutas principales usadas por el pipeline."""

    root: Path
    data: Path
    raw_data: Path
    masks: Path
    processed_data: Path
    checkpoints: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data=root / "data",
            raw_data=root / "data" / "raw",
            masks=root / "data" / "masks",
            processed_data=root / "data" / "processed",
            checkpoints=root / "models" / "checkpoints",
            reports=root / "reports",
        )


PATHS = ProjectPaths.from_root(PROJECT_ROOT)
