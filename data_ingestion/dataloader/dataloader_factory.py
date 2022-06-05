from torch.utils.data import DataLoader

from .utils import load_dataset
from data_ingestion.dataset.utils import random_split


__all__ = ["get_dataloaders"]


def get_dataloaders(cfg):
    """Retrieves a train and a validation DataLoader."""
    if cfg["mode"] == "standard":
        return get_standard_dataloaders(cfg["train_config"])



def get_standard_dataloaders(train_config):
    dataset = load_dataset(mode="standard", train_config=train_config)
    train_ratio = train_config["train_ratio"]
    test_ratio = train_config["test_ratio"]
    dataset_split = random_split(dataset, train_ratio, test_ratio)

    # train dataloader
    train_dataset = dataset_split["train"]
    validation_dataset = dataset_split["validation"]
    test_dataset = dataset_split["test"]
    # train_sampler = get_sampler(kind="standard", dataset=train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=train_config["parameters"]["num_workers"],
        batch_size=train_config["parameters"]["batch_size"],
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=train_config["parameters"]["num_workers"],
        batch_size=train_config["parameters"]["batch_size"],
    )
    test_dataset = DataLoader(
        dataset=test_dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=train_config["parameters"]["num_workers"],
        batch_size=train_config["parameters"]["batch_size"],
    )

    full_dataset = DataLoader(
        dataset=dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=train_config["parameters"]["num_workers"],
        batch_size=train_config["parameters"]["batch_size"],
    )

    return {
        "train": train_dataloader,
        "validation": validation_dataloader,
        "test": test_dataset,
        "full": full_dataset,
    }



