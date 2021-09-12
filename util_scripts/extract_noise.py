import argparse
import os

import PIL
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    def __init__(self, source_dir: str, transform=None):
        self.main_dir = source_dir
        self.transform = transform
        self.do_transforms = transform is not None
        self.total_imgs = os.listdir(source_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PIL.Image.open(img_loc).convert("RGB")
        if self.do_transforms:
            image = self.transform(image)
        return image


def extract(source_dir: str, dest_dir: str, noise_level: int, window_size: int, kernel_size: int,
            operation_type: str) -> None:
    images = ImageFolderDataset(source_dir=source_dir,
                                transform=np.array)

    for i, img in tqdm(enumerate(images), total=len(images)):
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, noise_level, noise_level, window_size, window_size * 3
        )
        if operation_type == "noise":
            extracted_noise = img.astype(np.float32) - denoised.astype(np.float32)
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
                extracted_noise -= cv2.filter2D(extracted_noise, -1, kernel)

            extracted_noise -= np.mean(extracted_noise)
            cv2.imwrite(
                dest_dir + f'{i:06}_s{noise_level:02}w{window_size:02}k{kernel_size:02}.png',
                cv2.cvtColor(np.round(extracted_noise + 128).astype(np.uint8), cv2.COLOR_RGB2BGR))
        elif operation_type == "denoise":
            cv2.imwrite(
                dest_dir + f"{i:06}_denoised_s{noise_level:02}w{window_size:02}k{kernel_size:02}.png",
                cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unknown operation")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source_dir", type=str, default=None)
    argparser.add_argument("--dest_dir", type=str, default=None)
    argparser.add_argument("--noise_level", type=int, default=None)
    argparser.add_argument("--window_size", type=int, default=7)
    argparser.add_argument("--blur_kernel_size", type=int, default=0)
    argparser.add_argument("--operation", type=str, default="noise")
    args = argparser.parse_args()

    if args.source_dir is None or args.dest_dir is None:
        raise ValueError("Both source dir and destination dir must be specified.")

    if args.noise_level is None:
        raise ValueError("Noise level must be specified.")

    if args.dest_dir[-1] != "/":
        args.dest_dir += "/"
    extract(args.source_dir, args.dest_dir, args.noise_level, args.window_size, args.blur_kernel_size, args.operation)
