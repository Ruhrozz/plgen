import warnings

from plgen.dataset.aspect_ratio_datamodule import AspectRatioDataModule
from plgen.dataset.imagecrop_datamodule import ImageCropDataModule

datamodule_registry = {
    "image_crop": ImageCropDataModule,
    "aspect_ratio": AspectRatioDataModule,
}


def get_datamodule(cfg):
    if cfg.dataset.type == "aspect_ratio":
        warnings.warn("AspectRatioDataModule is not fully tested and implemented.")

    return datamodule_registry[cfg.dataset.type](cfg)
