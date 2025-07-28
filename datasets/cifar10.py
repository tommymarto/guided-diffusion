import os
import tempfile

import torchvision
import numpy as np
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    for split in ["train", "test"]:
        base_dir = "/nfs/ghome/live/martorellat/data"
        out_dir = os.path.join(base_dir, f"cifar_{split}")
        npz_path = os.path.join(base_dir, f"images_{split}.npz")
        # if os.path.exists(out_dir):
        #     print(f"skipping split {split} since {out_dir} already exists.")
        #     continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images...")
        # os.mkdir(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        images = []
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)
            images.append(image)
        # save images as npz file
        images_np = np.stack([np.array(img) for img in images])
        np.savez(npz_path, images_np)



if __name__ == "__main__":
    main()