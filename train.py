import pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

from dl_module.callbacks import get_checkpont_callback
from dl_module.dataset import get_train_val_ds, get_test_ds, get_inference_ds
from dl_module.loader import get_train_val_loader, get_test_loader, get_inference_loader
from dl_module.loss import get_criterion
from dl_module.model import get_model
from dl_module.scheduler import get_optimizer_and_scheduler
from dl_module.srmodule import LitSuperResolutionModule
from dl_module.transform import get_train_val_tfms, get_test_tfms, get_inference_tfms
from inference import inference
from settings import DEVICE
from utils.config import Config
from utils.parser import get_train_parser

if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()

    cfg = Config(args.config)

    if hasattr(cfg.general, "seed"):
        pytorch_lightning.seed_everything(cfg.general.seed, workers=True)

    train_tfms, train_noise_tfms, val_tfms, val_noise_tfms = get_train_val_tfms(cfg)
    train_ds, val_ds = get_train_val_ds(cfg,
                                        train_tfms, train_noise_tfms,
                                        val_tfms, val_noise_tfms)

    train_loader, val_loader = get_train_val_loader(cfg, train_ds, val_ds)

    sr_model = get_model(cfg)

    criterion = get_criterion(cfg)

    optimizer, scheduler = get_optimizer_and_scheduler(cfg, sr_model, train_loader)

    checkpoint_callback = get_checkpont_callback(cfg)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    test_loader, log_frequency = None, None
    if hasattr(cfg, "test"):
        test_tfms = get_test_tfms(cfg)
        test_ds = get_test_ds(cfg, test_tfms)
        test_loader = get_test_loader(cfg, test_ds)
        log_frequency = cfg.test.log_frequency

    litmodel = LitSuperResolutionModule(scale=cfg.general.scale,
                                        sr_model=sr_model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        val_img_log_count=cfg.logging.image_log_count,
                                        log_metrics=cfg.logging.log_metrics,
                                        test_loader=test_loader,
                                        log_frequency=log_frequency)

    callbacks = [lr_monitor]
    if hasattr(cfg, "validation"):
        callbacks.append(checkpoint_callback)

    resume_checkpoint = None
    if hasattr(cfg.training, "resume_checkpoint"):
        resume_checkpoint = cfg.training.resume_checkpoint

    trainer = pl.Trainer(gpus=[0], max_epochs=cfg.general.epochs,
                         logger=pl_loggers.TensorBoardLogger(save_dir=cfg.logging.run_dir,
                                                             default_hp_metric=False),
                         deterministic=True,
                         log_every_n_steps=cfg.logging.log_every_n_steps,
                         callbacks=callbacks,
                         resume_from_checkpoint=resume_checkpoint)
    trainer.fit(litmodel, train_loader, val_loader)

    trainer.test(dataloaders=test_loader, ckpt_path="best")

if hasattr(cfg, "inference"):
    inference_tfms = get_inference_tfms(cfg)
    inference_ds = get_inference_ds(cfg, inference_tfms)
    inference_loader = get_inference_loader(cfg, inference_ds)

    sr_model = litmodel.sr_model
    sr_model.to(DEVICE)
    inference(sr_model, inference_loader, cfg.inference.out_dir)
