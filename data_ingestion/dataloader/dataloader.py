from configparser import ConfigParser
from typing import Tuple, Callable, Optional, List

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    WeightedRandomSampler,
    Subset,
)

from data_ingestion.samplers import get_sampler
from data_ingestion.dataset import SpoofDataset
from data_ingestion.dataset import TripletSpoofDataset
from data_ingestion.dataset import SpoofSiameseDataset


config = ConfigParser()
config.read("config.ini")
dataset_names = config["dataset"]["dataset_names"].split(", ")

Dataset = torch.utils.data.Dataset
data_loaders = Tuple[DataLoader, DataLoader]


# def get_triplet_dataloaders(batch_size_train: int = 4,
#                             batch_size_val: int = 32,
#                             additional_transform: Optional[Callable] = None,
#                             blacken_background: bool = False) -> data_loaders:
#     train_set = TripletSpoofDataset(mode='train',
#                                     extract_face=True,
#                                     blacken_background=blacken_background,
#                                     additional_transform=additional_transform)
#     val_set = TripletSpoofDataset(mode='val',
#                                   extract_face=True,
#                                   blacken_background=blacken_background,
#                                   additional_transform=additional_transform)
#     train_loader = DataLoader(train_set,
#                               batch_size=batch_size_train,
#                               num_workers=0)
#     val_loader = DataLoader(val_set,
#                             batch_size=batch_size_val,
#                             num_workers=0)
#     return train_loader, val_loader


def get_dataloaders(
    batch_size: int = 256,
    dataset_kind: str = "standard",
    additional_transform: Optional[Callable] = None,
    blacken_background: bool = False,
) -> data_loaders:
    """Retrieves train and validation dataloaders. These are a constructed with
    the union of all datasets. A weighted random sampler is implemented to
    avoid dataset and class bias
    Args:
        dataset_kind: One of {'standard', 'siamese', 'triplet'}
    """
    dataset = load_dataset(
        dataset_kind=dataset_kind,
        additional_transform=additional_transform,
        blacken_background=blacken_background,
    )
    sampler = get_sampler(kind=dataset_kind, dataset=dataset)

    train_indices, val_indices = split_idx(max_index=len(dataset))
    train_dataset, val_dataset = split_dataset(
        dataset, train_indices, val_indices
    )
    train_sampler, _ = split_sampler(sampler, train_indices, val_indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        num_workers=0,
        batch_size=batch_size,
    )

    validation_loader = DataLoader(
        dataset=val_dataset, shuffle=True, num_workers=0, batch_size=batch_size
    )

    return train_loader, validation_loader


# place this split functions somewhere in the utils123 directory
def split_idx(
    max_index: int, train_size: float = 0.8
) -> Tuple[List[int], List[int]]:
    indices = torch.randperm(max_index).tolist()
    split_point = int(max_index * train_size)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    return train_indices, val_indices


def split_dataset(
    dataset: Dataset, train_indices: List[int], val_indices: List[int]
) -> Tuple[Dataset, Dataset]:
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    return train_set, val_set


def split_sampler(sampler, train_indices, val_indices):
    weights = sampler.weights
    train_sampler = WeightedRandomSampler(
        weights=weights[train_indices], num_samples=len(weights)
    )
    val_sampler = WeightedRandomSampler(
        weights=weights[val_indices], num_samples=len(weights)
    )
    return train_sampler, val_sampler


#  add a dataset factory ?
def load_dataset(
    dataset_kind: str,
    additional_transform: Optional[Callable] = None,
    blacken_background: bool = False,
) -> List[Dataset]:
    if dataset_kind == "standard":
        return load_standard_dataset(
            additional_transform=additional_transform,
            blacken_background=blacken_background,
        )
    elif dataset_kind == "siamese":
        return SpoofSiameseDataset(
            extract_face=True,
            blacken_background=blacken_background,
            additional_transform=additional_transform,
        )
    elif dataset_kind == "triplet":
        return TripletSpoofDataset(
            extract_face=True,
            blacken_background=blacken_background,
            additional_transform=additional_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def load_standard_dataset(
    additional_transform: Optional[Callable] = None,
    blacken_background: bool = False,
) -> Dataset:
    datasets = []
    for name in dataset_names:
        if name == "CASIA":
            dataset = SpoofDataset(
                dataset_name=name,
                extract_face=False,
                blacken_background=False,
                additional_transform=additional_transform,
            )
        else:
            dataset = SpoofDataset(
                dataset_name=name,
                extract_face=True,
                blacken_background=blacken_background,
                additional_transform=additional_transform,
            )
        datasets.append(dataset)
    return ConcatDataset(datasets)
