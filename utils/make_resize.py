import argparse
import os
import pathlib
from glob import glob

from PIL import Image
from rich.progress import track


def _center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Source data path")
    parser.add_argument("destination", type=str, help="Destination for resized data")
    parser.add_argument(
        "--img_size",
        type=int,
        help="Length of a square side for every image",
        default=512,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = glob(args.source + "/*.jpg")

    os.makedirs(args.destination, exist_ok=True)

    for image_path in track(images):
        pathlib_image = pathlib.Path(image_path)
        os.system(
            f"cp {str(pathlib_image.parent / pathlib_image.stem)}.txt {args.destination}",
        )

        img = Image.open(image_path)
        img = _center_crop(img)
        img = img.resize(
            (args.img_size, args.img_size),
            resample=Image.LANCZOS,
        )
        img.save(args.destination + f"/{pathlib_image.name}")


if __name__ == "__main__":
    main()
