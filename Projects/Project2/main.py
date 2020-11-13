import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Sequential, Linear, ReLU


class DeepDTAModified(nn.Module):
    def __init__(self, datatype, dim, layer_cnn, layer_output):
        super(DeepDTAModified, self).__init__()

        if datatype == 'davis':
            self.SEQLEN = 1200
            self.SMILEN = 85
        elif datatype == 'kiba':
            self.SEQLEN = 1000
            self.SMILEN = 100

        self.embed_word = nn.Embedding(CHARPROTLEN, dim)
        self.embed_smile = nn.Embedding(CHARCANSMILEN, dim)

        self.layer_cnn = layer_cnn
        self.layer_output = layer_output
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1 if i == 0 else 2 ** (i + 1),
                                              out_channels=2 ** (i + 2),
                                              kernel_size=2 * (i + 2) + 1,
                                              stride=4,
                                              padding=i + 2) for i in range(layer_cnn)])

        self.W_rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=dim, hidden_size=dim)

        # self.W_attention = nn.Linear(dim, 100)
        # self.P_attention = nn.Linear(100, 100)
        cnn_params = int((self.SEQLEN * dim) * (2 ** (layer_cnn + 1) / (16 ** layer_cnn))) + 8
        rnn_params = int(2 * self.SMILEN * dim)
        # print(cnn_params)
        # print(rnn_params)
        self.W_out = nn.ModuleList(
            [nn.Linear(int((cnn_params + rnn_params) / (i + 1)), int((cnn_params + rnn_params) / (i + 2)))
             for i in range(layer_output)])
        # print(int((cnn_params + rnn_params) / 1))
        self.W_interaction = nn.Linear(int((cnn_params + rnn_params) / (layer_output + 1)), 1)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def attention_cnn(self, xs):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(xs, 1)
        # xs = xs.permute(0, 2, 1)
        for i in range(self.layer_cnn):
            # print(xs.shape)
            xs = self.W_cnn[i](xs)
            xs = torch.relu(xs)
        # print("!!!!!!!!!!!!!!!!!!!!")
        # print(xs.shape)
        # print("!!!!!!!!!!!!!!!!!!!!")
        # xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        # print("????????????????????")
        # print(xs.shape)
        # print("????????????????????")
        # xs = torch.unsqueeze(torch.mean(xs, 0), 0)
        # print("###################")
        return torch.flatten(xs, start_dim=1)

    def rnn(self, xs):
        # xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        # xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        # xs = torch.unsqueeze(torch.mean(xs, 0), 0)
        # print(xs.shape)
        return torch.flatten(xs, start_dim=1)

    def forward(self, inputs):
        words, smiles = inputs

        # print(words.shape)
        # print(smiles.shape)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)

        # print(word_vectors.shape)

        protein_vector = self.attention_cnn(word_vectors)

        # print(protein_vector.shape)

        """smile vector with attention-CNN."""
        # add the feature of word embedding of SMILES string
        smile_vectors = self.embed_smile(smiles)

        # print(smile_vectors.shape)

        after_smile_vectors = self.rnn(smile_vectors)

        # print(after_smile_vectors.shape)

        """Concatenate the above two vectors and output the interaction."""
        # concatenate with three types of features

        cat_vector = torch.cat((protein_vector, after_smile_vectors), 1)
        cat_vector = torch.squeeze(cat_vector)

        # print(cat_vector.shape)

        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
            # print(j)
        interaction = self.W_interaction(cat_vector)

        return interaction

the_model = DeepDTAModified('davis', 128, 3, 1)
the_model.load_state_dict(torch.load('model_davis_normal.pt'))

