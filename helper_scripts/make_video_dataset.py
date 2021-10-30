import cv2
from tqdm import tqdm

from utils.parser import get_make_film_dataset_parser

if __name__ == "__main__":
    argparser = get_make_film_dataset_parser()
    args = argparser.parse_args()

    source_path, dest_dir, frequency, prefix = args.source_path, args.dest_dir, args.frequency, args.prefix

    if source_path is None or frequency is None:
        raise ValueError("Both film file path and destination folder must be specified.")

    if frequency is None:
        raise ValueError("Sampling frequency must be specified.")

    if dest_dir[-1] != "/":
        dest_dir += "/"

    cap = cv2.VideoCapture(source_path)

    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Frames"):
        suc, img = cap.read()
        if not suc:
            raise Exception("Couldn't read frame:", i)
        if i % frequency == 0:
            cv2.imwrite(dest_dir + prefix + f"{i:06}.png", img)
