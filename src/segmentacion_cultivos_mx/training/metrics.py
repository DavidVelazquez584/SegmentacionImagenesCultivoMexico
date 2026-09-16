"""Métricas base para evaluar segmentación."""


def binary_iou(y_true: list[int], y_pred: list[int]) -> float:
    """Calcula IoU binario usando listas simples de 0 y 1."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud.")

    intersection = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    union = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 or pred == 1)

    if union == 0:
        return 1.0
    return intersection / union
