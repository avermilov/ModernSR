from torch.utils.data import DataLoader, Dataset

from utils.config import Config


def get_train_val_loader(cfg: Config, train_ds: Dataset, val_ds: Dataset) -> (DataLoader, DataLoader):
    train_batch_size = cfg.training.batch_size
    train_workers = cfg.training.workers
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, num_workers=train_workers, shuffle=True)

    val_loader = None
    if hasattr(cfg, "validation"):
        val_batch_size = cfg.validation.batch_size
        val_workers = cfg.validation.workers
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=val_workers, shuffle=False)

    return train_loader, val_loader


def get_test_loader(cfg: Config, test_ds: Dataset) -> DataLoader:
    test_workers = cfg.test.workers
    test_batch_size = cfg.test.batch_size

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=test_workers, shuffle=False)

    return test_loader


def get_inference_loader(cfg: Config, test_ds: Dataset) -> DataLoader:
    inference_workers = cfg.inference.workers
    inference_batch_size = cfg.inference.batch_size

    inference_loader = DataLoader(test_ds, batch_size=inference_batch_size, num_workers=inference_workers,
                                  shuffle=False)

    return inference_loader
