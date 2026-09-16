"""Script base para entrenar el modelo de segmentación."""

from pathlib import Path

from segmentacion_cultivos_mx.training.train import run_training


def main() -> None:
    run_training(config_path=Path("configs/base.yaml"))


if __name__ == "__main__":
    main()
