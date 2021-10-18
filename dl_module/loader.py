from typing import Dict

from torch.utils.data import DataLoader, Dataset


def get_train_val_loader(d: Dict, train_ds: Dataset, val_ds: Dataset) -> (DataLoader, DataLoader):
    validate = d["validate"]
    d = d["loaders"]

    train_batch_size = d["train_batch_size"]
    train_workers = d["train_workers"]
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, num_workers=train_workers)

    val_loader = None
    if validate:
        val_batch_size = d["validation_batch_size"]
        val_workers = d["validation_workers"]
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=val_workers)

    return train_loader, val_loader


def get_test_loader(d: Dict, test_ds: Dataset) -> DataLoader:
    d = d["loaders"]

    test_workers = d["test_workers"]
    test_batch_size = d["test_batch_size"]

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=test_workers)

    return test_loader
