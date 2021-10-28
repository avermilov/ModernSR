import torchvision.transforms as tfms

from utils.config import Config


def get_train_val_tfms(cfg: Config) -> (tfms.Compose, tfms.Compose, tfms.Compose, tfms.Compose):
    scale = cfg.general.scale

    train_crop = cfg.training.crop
    train_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tfms.RandomHorizontalFlip(),
        tfms.RandomVerticalFlip(),
        tfms.RandomCrop(train_crop, train_crop)
    ])

    train_noise_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tfms.RandomCrop(train_crop // scale, train_crop // scale)
    ])

    val_tfms, val_noise_tfms = None, None
    if hasattr(cfg, "validation"):
        validation_crop = cfg.validation.crop
        val_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            tfms.CenterCrop(validation_crop)
        ])

        val_noise_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            tfms.RandomCrop(validation_crop // scale, validation_crop // scale)
        ])

    return train_tfms, train_noise_tfms, val_tfms, val_noise_tfms


def get_test_tfms(cfg: Config) -> tfms.Compose:
    test_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return test_tfms

def get_inference_tfms(cfg: Config) -> tfms.Compose:
    return get_test_tfms(cfg)
