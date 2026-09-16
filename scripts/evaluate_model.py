"""Script base para evaluar un modelo entrenado."""

from segmentacion_cultivos_mx.evaluation.evaluate import run_evaluation


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
