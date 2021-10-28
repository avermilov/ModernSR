import torch.nn as nn

from model_zoo import rdn, rrdb, srresnet, srcnn
from utils.config import Config


def get_model(cfg: Config) -> nn.Module:
    scale = cfg.general.scale

    model_type = cfg.model.type.lower()
    if model_type == "srcnn":
        return srcnn.SRCNN(num_channels=3, scale=scale)
    elif model_type == "srresnet":
        return srresnet.Generator(upscale_factor=scale)
    elif model_type == "rdn":
        return rdn.RDN(scale_factor=scale, blurpool=False, num_channels=3, num_features=64,
                       growth_rate=64, num_blocks=16, num_layers=8)
    elif model_type == "rdn_blurpool":
        return rdn.RDN(scale_factor=scale, blurpool=True, num_channels=3, num_features=64,
                       growth_rate=64, num_blocks=16, num_layers=8)
    elif model_type.startswith("rrdb"):
        num_rrdb_blocks = int(model_type[model_type.find("b") + 1:])
        return rrdb.Generator(scale=scale, num_rrdb_blocks=num_rrdb_blocks)
    else:
        raise ValueError("Unknown model type. Supported models are 'SRCNN', 'SRRESNET', 'RDN' and 'RRDB'")
