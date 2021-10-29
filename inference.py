import warnings

from torch import nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

import os
from os import path
import torch
import torchvision.utils
from tqdm import tqdm

from dl_module.dataset import get_inference_ds
from dl_module.loader import get_inference_loader
from dl_module.model import get_model
from dl_module.transform import get_inference_tfms
from utils.config import Config
from utils.parser import get_inference_parser
from settings import DEVICE


def inference(sr_model: nn.Module, inference_loader: DataLoader, out_dir: str):
    sr_model.eval()
    total_processed = 0
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Inference"):
            batch = batch.to(DEVICE)
            pred = sr_model(batch)
            pred = torch.clamp((pred + 1) / 2, min=0, max=1)

            for i in range(pred.shape[0]):
                torchvision.utils.save_image(pred[i],
                                             os.path.join(out_dir, f"image_{total_processed:05}.png"))

                total_processed += 1


if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()

    if args.config is None:
        cfg = object()
        cfg.general = object()
        cfg.inference = object()
        cfg.general.model = args.model
        cfg.general.scale = args.scale
        cfg.inference.in_dir = args.in_dir
        cfg.inference.out_dir = args.out_dir
        cfg.inference.checkpoint_path = args.checkpoint_path
        cfg.inference.batch_size = args.batch_size
        cfg.inference.workers = args.workers
    else:
        cfg = Config(args.config)

    assert path.exists(cfg.inference.in_dir), "Input source directory does not exist."
    assert path.exists(cfg.inference.checkpoint_path), "Checkpoint path does not exist."
    if not path.exists(cfg.inference.out_dir):
        os.makedirs(cfg.inference.out_dir)

    sr_model = get_model(cfg)
    model_weights = torch.load(cfg.inference.checkpoint_path)["state_dict"]
    # Hack for loading weights
    model_weights = {key.replace("sr_model.", ""): value for key, value in model_weights.items()}
    sr_model.load_state_dict(model_weights, strict=False)
    sr_model.to(DEVICE)

    inference_tfms = get_inference_tfms(cfg)
    inference_ds = get_inference_ds(cfg, inference_tfms)
    inference_loader = get_inference_loader(cfg, inference_ds)

    inference(sr_model, inference_loader, cfg.inference.out_dir)
