from typing import Dict

import pandas as pd
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import default_collate
from data_ingestion.dataset.dataset_factory import get_dataset


def load_dataset(mode, train_config):
    """Retrieve a single dataset which is the concatenation of all the datasets
    defined in the configuration file for a specific dataset kind
    (train | validation).
    """
    metas = []
    datasets = []
    for dataset_name in train_config["datasets"]:
        dataset_config = train_config["datasets"][dataset_name]
        dataset = get_dataset(dataset_mode=mode, dataset_config=dataset_config)
        datasets.append(dataset)
        metas.append(dataset.meta)
    concatenated_dataset = ConcatDataset(datasets)
    concatenated_dataset.meta = pd.concat(metas, ignore_index=True)
    return concatenated_dataset


def standard_collate(batch):
    """Same as PyTorch ´default_collate´ but filters None values for
    dictionaries."""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Dict):
        collated = {}
        for key in elem:
            values = [d[key] for d in batch if d[key] is not None]
            collated[key] = default_collate(values)
        return collated
    else:
        default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, "
            "numbers, dicts or lists; found {}"
        )
    raise TypeError(default_collate_err_msg_format.format(elem_type))
