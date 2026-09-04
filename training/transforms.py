"""Transformations préservant le ratio des images CastagNet."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResizeWithPad:
    """Redimensionne le côté long puis centre l'image dans un carré noir."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = F.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        horizontal = self.size - new_width
        vertical = self.size - new_height
        padding = [horizontal // 2, vertical // 2, horizontal - horizontal // 2, vertical - vertical // 2]
        return F.pad(resized, padding, fill=0)


def build_transforms(input_size: int, train: bool):
    operations = [ResizeWithPad(input_size)]
    if train:
        # L'orientation du fruit peut varier ; ces transformations restent faibles et géométriques.
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR, fill=0),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(operations)
