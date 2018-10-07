"""!
@brief Simple LSTM encoder for embedding the input using a simple
LSTM architecture

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import torch
import torch.nn as nn

class BLSTMEncoder(nn.Module):
    def __init__(self,
                 n_timesteps=250,
                 n_features=257,
                 num_layers=1,
                 hidden_size=None,
                 embedding_depth=None,
                 bidirectional=True):
        super(BLSTMEncoder, self).__init__()

        if n_timesteps is None or n_features is None:
            raise ValueError("You have to define both the number of "
                             "timesteps in each sequence and the "
                             "number of features for each timestep.")
        else:
            self.emb_dim = n_features * embedding_depth

        self.embedding_depth = embedding_depth
        self.hidden_size = hidden_size
        self.n_timesteps = n_timesteps
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        # assert len(self.hidden_sizes) == num_layers, 'Each layer ' \
        #        'should be defined by a corresponding hidden size.'
        self.rnn = nn.LSTM(input_size=n_features,
                           num_layers=num_layers,
                           hidden_size=self.hidden_size,
                           bidirectional=bidirectional,
                           batch_first=True)
        self.affine = nn.Linear(self.n_directions*self.hidden_size,
                                self.emb_dim)

    def forward(self, x):
        rnn_out, (hidden, states) = self.rnn(x)
        nonl_embedding = torch.sigmoid(self.affine(rnn_out))
        v = nonl_embedding.contiguous().view(x.size(0),
                                             -1,
                                             self.embedding_depth)
        return nn.functional.normalize(v, dim=-1, p=2)


if __name__ == "__main__":

    model = BLSTMEncoder()
