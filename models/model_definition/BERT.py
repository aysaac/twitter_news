import torch
import torch.nn as nn
from transformers import BertModel
class SentimentClassifier(nn.Module):#modelos de clasisifcacion

    def __init__(self, n_classes,path,droputprob):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(path,return_dict=False)#se carga el modelo desde path
        self.drop = nn.Dropout(p=droputprob)#droput para entrenar
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)#output lineal final

    def forward(self, input_ids, attention_mask):#foward prop
        _, pooled_output = self.bert(
            input_ids=input_ids,#input
            attention_mask=attention_mask#mask
        )
        output = self.drop(pooled_output)#output
        return self.out(output)