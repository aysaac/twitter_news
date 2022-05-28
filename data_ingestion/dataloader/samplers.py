import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler


Sampler = WeightedRandomSampler
Dataset = torch.utils.data.Dataset


def get_sampler(kind: str, dataset: torch.utils.data.Dataset) -> Sampler:
    """Retrieves a sampler...
    :param kind: One of {standard, fusion}
    :param dataset: Dataset for which to create the sampler.
    """
    if kind == "standard":
        return get_standard_sampler(dataset=dataset)
    elif kind == "fusion":
        return get_standard_sampler(dataset=dataset)
    else:
        raise ValueError(f"Unsupported kind: {kind}")


def get_standard_sampler(dataset):
    datasets = dataset.datasets  # Dataset is a concatenation of datasets
    # Weights for balanced datasets
    lengths = [len(dataset) for dataset in datasets]
    d_weights = [np.ones((length), dtype=float) / length for length in lengths]
    d_weights = np.concatenate(d_weights)
    # Class balance weights
    all_labels = []
    for dataset in datasets:
        paths, labels = zip(*dataset._image_list)
        all_labels.extend(labels)
    labels = np.array(all_labels)
    c_weights = np.where(
        (labels == 1), (1 / (labels == 1).sum()), (1 / (labels == 0).sum())
    )
    weights = d_weights * np.array(c_weights)
    return WeightedRandomSampler(weights, len(weights))


# def get_siamese_sampler(dataset: Dataset) -> Sampler:
#     paths, labels = zip(*dataset._image_list)
#     # Weights for balanced datasets
#     counts = dataset._dataset_counts  # dict -> dataset_name:dataset_counts
#     d_weights = [1 / v for k, v in counts.items() for _ in range(counts[k])]
#     # Weights balanced classes
#     labels = np.array(labels)
#     c_weights = np.where((labels == 1),
#                          (1 / (labels == 1).sum()),
#                          (1 / (labels == 0).sum()))
#     # Weights for balanced datasets and classes
#     weights = np.array(c_weights) * np.array(d_weights)
#     return WeightedRandomSampler(weights, len(weights))


# def get_triplet_sampler(dataset: Dataset) -> Sampler:
#     paths, labels = zip(*dataset._image_list)
#     # Weights for balanced datasets
#     counts = dataset._dataset_counts  # dict -> dataset_name:dataset_counts
#     d_weights = [1 / v for k, v in counts.items() for _ in range(counts[k])]
#     # Weights balanced classes
#     labels = np.array(labels)
#     c_weights = np.where((labels == 1),
#                          (1 / (labels == 1).sum()),
#                          (1 / (labels == 0).sum()))
#     # Weights for balanced datasets and classes
#     weights = np.array(c_weights) * np.array(d_weights)
#     return WeightedRandomSampler(weights, 3 * len(weights))
