from random import randrange

import lightning as L
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ImageCropDataModule(L.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.dataset = None

    def setup(self, stage: str):
        if stage == "fit":
            self.dataset = ImageCropDataset(
                images_paths_csv=self.conf.dataset.data_csv,
                img_size=self.conf.dataset.image_size,
                work_dir=self.conf.dataset.work_dir,
            )

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("Dataset is not created yet! Run `self.setup`")

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.conf.dataloader.batch_size,
            num_workers=self.conf.dataloader.num_workers,
        )
        return dataloader


class ImageCropDataset(Dataset):
    def __init__(
        self,
        images_paths_csv,
        img_size,
        work_dir,
    ):
        self.img_size = img_size
        self.work_dir = work_dir + "/"
        csv = pd.read_csv(images_paths_csv)
        self.paths = csv["path"].values
        self.captions = csv["caption"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.work_dir + self.paths[i])
        img = _center_crop(img)
        # img = _random_crop(img)
        img = img.resize(
            (self.img_size, self.img_size),
            resample=Image.Resampling.LANCZOS,
            reducing_gap=1,
        )
        img = np.array(img.convert("RGB"))
        img = img.astype(np.float32) / 127.5 - 1
        return np.transpose(img, [2, 0, 1]), self.captions[i]


def _center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def _random_crop(image):
    width, height = image.size
    new_size = min(width, height)
    w1 = 0 if width == new_size else randrange(0, width - new_size)
    h1 = 0 if height == new_size else randrange(0, height - new_size)
    return image.crop((w1, h1, w1 + new_size, h1 + new_size))
