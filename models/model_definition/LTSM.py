import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self,input_size=300,dimension=128,bidirectional=True):
        super(LSTM, self).__init__()

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)
    def foward(self,x):
        pass