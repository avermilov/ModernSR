from utils.parser import get_split_parser
import os
import shutil
from random import shuffle

if __name__ == "__main__":
    parser = get_split_parser()
    args = parser.parse_args()

    src_dir, train_share = args.src_dir, args.train_share

    train_dir = os.path.join(src_dir, "train")
    valid_dir = os.path.join(src_dir, "valid")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    files = os.listdir(args.src_dir)
    files = [f for f in files if os.path.isfile(os.path.join(args.src_dir, f))]
    shuffle(files)
    train_size = int(len(files) * train_share)
    for f in files[:train_size]:
        shutil.move(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in files[train_size:]:
        shutil.move(os.path.join(src_dir, f), os.path.join(valid_dir, f))
