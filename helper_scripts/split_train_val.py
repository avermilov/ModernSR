import sys
from typing import List

sys.path.append("..")
from utils.parser import get_split_parser
import os
import shutil
from random import shuffle


def get_all_files(dir: str, rec: bool = False) -> List[str]:
    listOfFile = os.listdir(dir)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dir, entry)
        if os.path.isdir(fullPath):
            if rec:
                allFiles = allFiles + get_all_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


if __name__ == "__main__":
    parser = get_split_parser()
    args = parser.parse_args()

    src_dir, train_share, recursive = args.src_dir, args.train_share, args.recursive

    train_dir = os.path.join(src_dir, "train")
    valid_dir = os.path.join(src_dir, "valid")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    files = get_all_files(args.src_dir, recursive)
    shuffle(files)
    train_size = int(len(files) * train_share)
    for f in files[:train_size]:
        shutil.move(f, os.path.join(train_dir, f[f.rfind("/") + 1:]))
    for f in files[train_size:]:
        shutil.move(f, os.path.join(valid_dir, f[f.rfind("/") + 1:]))
