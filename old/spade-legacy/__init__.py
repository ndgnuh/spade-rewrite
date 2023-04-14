import torch
from torch import nn


class GraphGenerator(nn.Module):
    def __init__(self, config, fields):
        super().__init__()
        self.n_fields = len(fields)
        self.fields = fields

        h_size = config.hidden_size
        self.W_h = nn.Linear(h_size, h_size)
        self.W_d = nn.Linear(h_size, h_size)
        self.W_0 = nn.Linear(h_size, h_size)
        self.W_1 = nn.Linear(h_size, h_size)

    def forward(self, score):
        h_part_1 = score[:, :self.n_fields, :]
        h_part_2 = self.W_h(score[:, self.n_fields:, :])
        h = torch.cat([h_part_1, h_part_2], dim=1)
        d = self.W_d(score[:, self.n_fields:, :])
        s1 = torch.einsum('bih,bjh->bij', h, self.W_0(d))
        s0 = torch.einsum('bih,bjh->bij', h, self.W_1(d))
        es0 = torch.exp(s0)
        es1 = torch.exp(s1)
        p = es0 / (es0 + es1)
        return p
