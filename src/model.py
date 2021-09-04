import torch

import torch.nn as nn
from torch.autograd import Variable


class GenerativeMoleculesModel(nn.Module):
    def __init__(self, vocabs_size, hidden_size, output_size, embedding_dimension, n_layers, bidirectional=False):
        super(GenerativeMoleculesModel, self).__init__()
        self.vocabs_size = vocabs_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dimension = embedding_dimension
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocabs_size, embedding_dimension)
        self.rnn = nn.LSTM(embedding_dimension, hidden_size, n_layers, dropout=0.2, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.linear = nn.Linear(hidden_size * 2, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = self.embedding(input)
        output, hidden = self.rnn(input.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.bidirectional:
            hidden = (Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size)),
                      Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size)))
        else:
            hidden = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                      Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return hidden
