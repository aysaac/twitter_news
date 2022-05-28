from torch.utils.data import DataLoader

from .utils import load_dataset
from .samplers import get_sampler
from spoof_detection.data_ingestion.dataset.utils import random_split
from spoof_detection.data_ingestion.dataset.dataset_factory import get_dataset


__all__ = ["get_dataloaders"]


def get_dataloaders(cfg):
    """Retrieves a train and a validation DataLoader."""
    if cfg["mode"] == "standard":
        return get_standard_dataloaders(cfg["train_config"])
    elif cfg["mode"] == "depth_map":
        return get_depth_map_dataloaders(cfg["train_config"])
    else:
        raise NotImplementedError()


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



def get_fusion_dataloader(data_config):
    dataset = load_dataset(mode="fusion", data_config=data_config)
    sampler = get_sampler(kind="fusion", dataset=dataset)
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        num_workers=data_config["parameters"]["num_workers"],
        batch_size=data_config["parameters"]["batch_size"],
    )


def get_siamese_dataloader(data_config):
    dataset = get_dataset(
        dataset_mode="siamese", datasets_config=data_config["datasets"]
    )
    return DataLoader(
        dataset=dataset,
        num_workers=data_config["parameters"]["num_workers"],
        batch_size=data_config["parameters"]["batch_size"],
    )


def get_depth_map_dataloaders(data_config):
    dataset = load_dataset(mode="depth_map", train_config=data_config)
    train_ratio = data_config["train_ratio"]
    test_ratio = data_config["test_ratio"]
    dataset_split = random_split(dataset, train_ratio, test_ratio)

    # train dataloader
    train_dataset = dataset_split["train"]
    validation_dataset = dataset_split["validation"]
    test_dataset = dataset_split["test"]
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=data_config["parameters"]["num_workers"],
        batch_size=data_config["parameters"]["batch_size"],
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=data_config["parameters"]["num_workers"],
        batch_size=data_config["parameters"]["batch_size"],
    )
    test_dataset = DataLoader(
        dataset=test_dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=data_config["parameters"]["num_workers"],
        batch_size=data_config["parameters"]["batch_size"],
    )
    full_dataset = DataLoader(
        dataset=dataset,
        sampler=None,
        shuffle=True,
        drop_last=True,
        num_workers=data_config["parameters"]["num_workers"],
        batch_size=data_config["parameters"]["batch_size"],
    )

    return {
        "train": train_dataloader,
        "validation": validation_dataloader,
        "test": test_dataset,
        "full": full_dataset,
    }
