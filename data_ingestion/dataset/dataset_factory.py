import pandas as pd
from definitions import MAX_LEN

from data_ingestion.transforms import build_transform
from data_ingestion.dataset import BERT_dataset,SBW_dataset,spacy_dataset
__all__ = ["get_dataset"]


def get_dataset(dataset_mode, dataset_config=None, **kwargs):
    """Retrieves a dataset.
    :param dataset_mode: One of {standard, fusion, siamese}.
    :param dataset_config: Configuration of a particular dataset.
    :param datasets_config: Configuration of a set of datasets.
    """
    if dataset_mode == "spacy":
        return get_spacy_dataset(dataset_config)
    elif dataset_mode == "bert":
        return get_bert_dataset(dataset_config)
    elif dataset_mode == "SBW":
        return get_SBW_dataset(kwargs["datasets_config"])
    else:
        raise NotImplementedError()


def get_spacy_dataset(dataset_config):
    transforms = build_transform(dataset_config["transformations"])
    df=pd.read_csv(dataset_config["path"])
    return spacy_dataset(
        texts=df["texts"], targets=df["target"],max_len=MAX_LEN,transforms=transforms
    )


def get_bert_dataset(dataset_config):
    transforms = build_transform(dataset_config["transformations"])
    df = pd.read_csv(dataset_config["path"])

    return BERT_dataset(
        texts=df["texts"], targets=df["target"], max_len=MAX_LEN, transforms=transforms
    )


def get_SBW_dataset(dataset_config):
    dataset_source = []
    df = pd.read_csv(dataset_config["path"])
    transforms = build_transform(dataset_config["transformations"])
    return SBW_dataset(
        texts=df["texts"], targets=df["target"], max_len=MAX_LEN, transforms=transforms
    )


