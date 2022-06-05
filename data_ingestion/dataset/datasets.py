import logging
from random import choice, randint
from typing import Tuple, Callable, Optional, List, Union, Dict
import os
import torch
from torch import from_numpy
from torch import Tensor
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import spacy
import gensim
from transformers import BertTokenizer
from definitions import MAX_LEN
#%%


class text_dataset(Dataset):  # declara objeto clase Dataset

    def __init__(self, texts, targets, tokenizer,):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()



class spacy_dataset(text_dataset):  # declara objeto clase Dataset
    def __init__(self, texts, targets,  transforms):
        # super().__init__(texts, targets, max_len,transforms)
        self.texts = texts
        self.targets = targets

        self.tokenizer = spacy.load('es_core_news_sm')
        self.transforms=transforms

    def __getitem__(self, item):
        texts = str(self.texts.iloc[item])  # cargael ejemplo de la matriz
        if self.transforms:
            words=self.transforms(texts)


        words=self.tokenizer(words)
        target = self.targets[item]  # carga el label
        vector=[]

        for x in range(MAX_LEN):
            if x<(len(words)-1):
                vector.append(words[x].vector)
            else:
                vector.append(np.zeros(96))

        words=np.array(vector)

        words=from_numpy(words)
        return {  # dictionario con los enoucdigs
            'texts': texts,  # texto normal
            'embeddings':words,
            'targets': torch.tensor(target, dtype=torch.long)  # carga los labels
        }
class BERT_dataset(text_dataset):  # declara objeto clase Dataset
    def __init__(self, texts, targets, transforms,vocab_path):
        self.transforms=transforms
        self.texts = texts
        self.targets =targets
        self.tokenizer=BertTokenizer.from_pretrained(
            vocab_path,do_lower_case=False,pad_to_max_length=True)

    def __getitem__(self, item):

        texts = str(self.texts.iloc[item])  # cargael ejemplo de la matriz
        if self.transforms:
            texts=self.transforms(texts)
        target = self.targets[item]  # carga el label
        encoding = self.tokenizer.encode_plus(  # tokenizador configurado
            texts,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {  # dictionario con los enoucdigs
            'texts': texts,  # texto normal
            'input_ids': encoding['input_ids'].flatten(),  # texto encondificado (osea con numeros)
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)  # carga los labels
        }
class SBW_dataset(text_dataset):  # declara objeto clase Dataset
    def __init__(self, texts, targets, transforms):
        self.texts = texts
        self.targets = targets
  # longitud maxima para padding
        self.transforms=transforms
        glove = gensim.models.KeyedVectors.load_word2vec_format(
            r'B:\PycharmProjects\twitter_news\data-ingestion\vocabs\SBW-vectors-300-min5.txt')
        word2index = {token: token_index for token_index, token in enumerate(glove.index_to_key)}
        word_embeddings = [glove[key] for key in word2index.keys()]
        word2index["PADDING"] = len(word2index)
        word_embeddings.append(np.zeros(len(word_embeddings[0])))
        word2index["UNKNOWN"] = len(word2index)
        word_embeddings.append(np.random.uniform(-0.25, 0.25, len(word_embeddings[0])))
        word_embeddings = np.array(word_embeddings)
        word_embeddings = torch.FloatTensor(word_embeddings)
        self.word2index=word2index
        self.word2index_key = word2index.keys()
        self.index2embeding=word_embeddings
    def __getitem__(self, item):
        global indexes
        texts = str(self.texts.iloc[item])
        if self.transforms:
            words=self.transforms(texts)
        else:
            words=texts
        words=words.split()
        indexes = []
        for x in range(MAX_LEN):
            if x >= (len(words) - 1):
                index = self.word2index['PADDING']
                indexes.append(index)
            else:
                if words[x] in self.word2index_key:
                    index=self.word2index[words[x]]
                    indexes.append(index)
                else:
                    index = self.word2index['UNKNOWN']
                    indexes.append(index)
            embedding = self.index2embeding[indexes]


        target = self.targets[item]
        target=torch.tensor(target, dtype=torch.long)

        return {  # dictionario con los enoucdigs
            'text': texts,# texto normal
            'embeddings':embedding,
            'targets':target   # carga los labels
        }
