"""Definiciones base para pares de imagen y máscara."""

from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationSample:
    """Representa una imagen y su máscara de segmentación asociada."""

    image_path: Path
    mask_path: Path


def list_segmentation_samples(image_dir: Path, mask_dir: Path) -> list[SegmentationSample]:
    """Empareja imágenes y máscaras usando el nombre base del archivo."""

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }

    common_ids = sorted(images.keys() & masks.keys())
    return [SegmentationSample(images[item_id], masks[item_id]) for item_id in common_ids]
