import random
from typing import Dict
from random import shuffle
import pandas as pd
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from definitions import Experiment_SEED


def random_split(
    dataset: Dataset, train_ratio, test_ratio
) -> Dict[str, Dataset]:
    """Randomly splits a dataset into two (train and validation) using the
    given split_ratio.

    :param dataset: Dataset to split
    :type dataset: Dataset
    :param split_ratio: Determines the size of the resulting datasets, defaults
        to 0.8
    :type split_ratio: int, optional
    :return: A dictionary with the keys 'train' and 'test' with the
        corresponding datasets as values.
    :rtype: Dict[str, Dataset]
    """

    dataset_meta = dataset.meta
    dataset_length = len(dataset)
    dataset_meta = dataset_meta.drop_duplicates()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset_meta["path"],
        dataset_meta["target"],
        test_size=(1 - train_ratio),
        random_state=Experiment_SEED,
        stratify=dataset_meta["target"],
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_ratio,
        random_state=Experiment_SEED,
        stratify=y_test,
    )

    return {
        "train": Subset(dataset, indices=list(X_train.index)),
        "validation": Subset(dataset, indices=list(X_val.index)),
        "test": Subset(dataset, indices=list(X_test.index)),
    }
