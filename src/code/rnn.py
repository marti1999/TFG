import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split

class RNN(nn.Module):
    def __init__(self, embeddings, LSTM_dim, n_layers, bidirectional):
        super().__init__()

        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embeddings.shape[1], LSTM_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(LSTM_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        # input_x is expected to be of size (example length, batch size) however the axes are
        # flipped so we permute them to be the correct size.
        embedded = self.embedding(input_x.permute(1, 0))
        # embedded size = (example length, batch size, embedding dimensions)
        output, (hidden, cell) = self.lstm(embedded)
        # hidden size = (number of layers * number of directions, batch size, number of hidden units)
        output = self.dropout(hidden[-1])
        # output size = (batch size, number of hidden units)
        output = self.fc(output)
        # output size = (batch size, 1)
        output = self.sigmoid(output)

        return output