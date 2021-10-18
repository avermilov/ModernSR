from typing import Dict

from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpont_callback(d: Dict) -> ModelCheckpoint:
    model = d["model"]
    d = d["logging"]

    monitor_metric = d["save_metric"].lower()
    assert monitor_metric in ["psnr", "lpips_alex", "lpips_vgg"], "Unknown save metric."
    mode = "max" if monitor_metric == "psnr" else "min"
    monitor = "Validation Metrics/Epoch Wise/PSNR" if monitor_metric == "psnr" \
        else "Validation Metrics/Epoch Wise/LPIPS Alex" if monitor_metric == "lpips_alex" else "Validation Metrics/Epoch Wise/LPIPS VGG"
    save_name_format = model + "_{epoch:04d}_" + monitor_metric + "{" + monitor + ":.3f}"

    save_top_k = d["save_top_k"]

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=None,
        filename=save_name_format,
        save_top_k=save_top_k,
        mode=mode,
        auto_insert_metric_name=False
    )

    return checkpoint_callback
