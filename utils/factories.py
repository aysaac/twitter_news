import os
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim




def get_device(
    device_type: str, device_index: Optional[Union[int, Tuple[int, ...]]] = None
) -> torch.device:
    if device_type == "cpu":
        return torch.device(device_type)
    elif device_type == "cuda":
        if isinstance(device_index, int):
            return torch.device(device_type + f":{device_index}")
        elif isinstance(device_index, list):
            raise NotImplementedError()
        else:
            raise ValueError("ConfigError: Invalid device index")
    else:
        raise ValueError("ConfigError: Invalid device type")


def get_optimizer(cfg, model):
    optimizer_params = cfg["train_config"]["parameters"]["optimizer"]
    optimizer_kind = optimizer_params.pop("kind")
    if optimizer_kind == "adam":
        return optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_kind == "sgd":
        return optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_kind == "adamw":
        return optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise NotImplementedError()


def get_loss(loss):
    if loss == "logit":
        return nn.BCEWithLogitsLoss()
    elif loss=="cross_entropy":
        return nn.CrossEntropyLoss()