import cv2
import numpy as np
import pyprind
import skvideo.io as vio
import torch
import sys

sys.path.append("..")

from dl_module.model import get_model
from utils.parser import get_inference_video_parser
from settings import DEVICE


class Object(object):
    pass


def torch_to_frame(img: torch.tensor) -> np.array:
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    img = torch.squeeze(img, dim=0)
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
    parser = get_inference_video_parser()
    args = parser.parse_args()

    vid_path = args.video_path
    save_name = args.save_name
    net_type = args.net_type
    checkpoint_path = args.checkpoint_path
    scale = args.scale
    compression_level = args.compression_level

    cfg = Object()
    cfg.model = Object()
    cfg.model.type = net_type
    cfg.general = Object()
    cfg.general.scale = scale

    model = get_model(cfg).to(DEVICE)
    model_weights = torch.load(checkpoint_path)["state_dict"]
    # Hack for loading weights
    model_weights = {key.replace("sr_model.", ""): value for key, value in model_weights.items()}
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    model.to(DEVICE)

    vcsan = cv2.VideoCapture(vid_path)


    out = vio.FFmpegWriter(save_name + ".mp4", outputdict={
        '-vcodec': 'libx264',
        '-crf': str(compression_level),
        '-preset': 'veryslow'
    })
    out_bic = vio.FFmpegWriter(save_name + "bic.mp4", outputdict={
        '-vcodec': 'libx264',
        '-crf': str(compression_level),
        '-preset': 'veryslow'
    })

    total_frames = int(vcsan.get(cv2.CAP_PROP_FRAME_COUNT))
    progbar = pyprind.ProgBar(total_frames, title="Inference")

    suc, img = vcsan.read()
    while suc:
        with torch.no_grad():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
            image = torch.from_numpy(img.transpose(2, 0, 1)).type(torch.FloatTensor)
            image = (image - 0.5) / 0.5
            image = torch.clamp(image, -1, 1)
            image = torch.unsqueeze(image, dim=0).to(DEVICE)
            image.requires_grad = False

            sr = model(image)
            sr = torch_to_frame(sr)
            out.writeFrame(sr)
            progbar.update()

        suc, img = vcsan.read()

    out.close()
    vcsan.release()
