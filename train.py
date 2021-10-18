import pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import json
import argparse

from pytorch_lightning.callbacks import LearningRateMonitor

from dl_module.loss import get_loss
from dl_module.model import get_model
from dl_module.dataset import get_train_val_ds, get_test_ds
from dl_module.loader import get_train_val_loader, get_test_loader
from dl_module.transform import get_train_val_tfms, get_test_tfms
from dl_module.scheduler import get_optimizer_and_scheduler
from dl_module.srmodule import LitSuperResolutionModule
from dl_module.checkpoint import get_checkpont_callback


def get_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("json", type=str,
                           help="Parameter file for training Super Resolution models.")

    return argparser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with open(args.json, "r") as file:
        params = json.load(file)

    seed = params["seed"]
    if seed is not None:
        pytorch_lightning.seed_everything(seed, workers=True)

    train_tfms, train_noise_tfms, val_tfms, val_noise_tfms = get_train_val_tfms(params)
    train_ds, val_ds = get_train_val_ds(params,
                                        train_tfms, train_noise_tfms,
                                        val_tfms, val_noise_tfms)

    train_loader, val_loader = get_train_val_loader(params, train_ds, val_ds)

    sr_model = get_model(params)

    loss = get_loss(params)

    optimizer, scheduler = get_optimizer_and_scheduler(params, sr_model, train_loader)

    checkpoint_callback = get_checkpont_callback(params)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    litmodel = LitSuperResolutionModule(scale=params["scale"],
                                        sr_model=sr_model,
                                        criterion=loss,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        val_img_log_count=params["logging"]["val_img_log_count"],
                                        log_metrics=params["logging"]["log_metrics"],
                                        )

    trainer = pl.Trainer(gpus=[0], max_epochs=params["epochs"],
                         logger=pl_loggers.TensorBoardLogger(save_dir=params["logging"]["run_dir"],
                                                             default_hp_metric=False),
                         deterministic=True,
                         log_every_n_steps=1,
                         callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(litmodel, train_loader, val_loader)

    if params["test"]:
        test_tfms = get_test_tfms(params)
        test_ds = get_test_ds(params, test_tfms)
        test_loader = get_test_loader(params, test_ds)

        trainer.test(dataloaders=test_loader, ckpt_path="best")
