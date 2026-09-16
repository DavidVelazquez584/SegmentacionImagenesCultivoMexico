"""Utilidades para dividir datos en entrenamiento, validación y prueba."""

from random import Random
from typing import Sequence


def split_identifiers(
    identifiers: Sequence[str],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Divide identificadores manteniendo una semilla reproducible."""

    if train_ratio + validation_ratio >= 1:
        raise ValueError("La suma de train_ratio y validation_ratio debe ser menor que 1.")

    items = list(identifiers)
    Random(seed).shuffle(items)

    train_end = int(len(items) * train_ratio)
    validation_end = train_end + int(len(items) * validation_ratio)

    return {
        "train": items[:train_end],
        "validation": items[train_end:validation_end],
        "test": items[validation_end:],
    }
