from pytorch_lightning.callbacks import ModelCheckpoint
from settings import *
from utils.config import Config


def get_checkpont_callback(cfg: Config) -> ModelCheckpoint:
    model = cfg.model.type

    monitor_metric = cfg.logging.save_metric.lower()
    assert monitor_metric in ["psnr", "lpips_alex", "lpips_vgg"], "Unknown save metric."

    mode = "max" if monitor_metric == "psnr" else "min"
    monitor = PSNR_LOG_NAME if monitor_metric == "psnr" else LPIPS_ALEX_LOG_NAME \
        if monitor_metric == "lpips_alex" else LPIPS_VGG_LOG_NAME
    save_name_format = model + "_{epoch:04d}_" + monitor_metric + "{" + monitor + ":.3f}"

    save_top_k = cfg.logging.save_top_k

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=None,
        filename=save_name_format,
        save_top_k=save_top_k,
        mode=mode,
        auto_insert_metric_name=False,
        save_weights_only=True
    )

    return checkpoint_callback
