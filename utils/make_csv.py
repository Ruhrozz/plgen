import argparse
import pathlib
from datetime import datetime
from glob import glob

import pandas as pd
from rich.progress import track


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Source data path")
    return parser.parse_args()


def main():
    args = parse_args()
    p = pathlib.Path(args.source)

    df = []
    for image_path in track(sorted(glob(str(p / "*.jpg")))):
        with open((image_path[:-4] + ".txt"), "r") as file:
            file = file.read()
            df.append([str(image_path), file])

    print(f"Total data: {len(df)}")

    df = pd.DataFrame(df)
    df.columns = ["paths", "caption"]

    for img_p, cap in zip(df["paths"].values, df["caption"].values):
        if isinstance(cap, float):
            print(f"Corrupted data found, {img_p}")

    dtime = datetime.strftime(datetime.now(), "%d-%m-%y_%H-%M-%S")
    df.to_csv(dtime + "_anime.csv", index=False)


if __name__ == "__main__":
    main()
