import sys

sys.path.append("..")
import cv2
import pyprind

from utils.parser import get_concat_videos_parser

if __name__ == "__main__":
    parser = get_concat_videos_parser()
    args = parser.parse_args()

    vid1, vid2 = args.vid1, args.vid2

    if args.vid1 is None or args.vid2 is None:
        raise ValueError("Both video paths must be specified.")

    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)

    fps = cap1.get(cv2.CAP_PROP_FPS)

    vert_concat = args.stack == "v"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_resolution = (2 * w, h) if not vert_concat else (w, 2 * h)
    out = cv2.VideoWriter(vid1[max(vid1.rfind("/"), 0):vid1.rfind(".")] + '_and_' + vid2, fourcc, fps, new_resolution)

    total = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    iter_bar = pyprind.ProgBar(total, title='Concatenation', stream=sys.stdout)

    while True:
        suc1, frame1 = cap1.read()
        suc2, frame2 = cap2.read()
        if not suc1 or not suc2:
            break
        final = cv2.hconcat([frame1, frame2]) if not vert_concat else cv2.vconcat([frame1, frame2])
        out.write(final)
        iter_bar.update()

    out.release()
    cap1.release()
    cap2.release()
