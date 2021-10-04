from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def get_scheduler(d: Dict, model: nn.Module, dataloader: DataLoader = None):
    epochs = int(d["epochs"])
    d = d["loss"]
    learning_rate = float(d["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    scheduler_type = d["scheduler_type"].lower()
    if scheduler_type == "decay":
        gamma = float(d["gamma"])
        return optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=learning_rate,
                                                  epochs=epochs, steps_per_epoch=len(dataloader))
        # Hack because OneCycleLR requires updating every step, not epoch
        return {"scheduler": scheduler, "interval": "step"}
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}.")
