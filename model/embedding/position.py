import torch.nn as nn
import torch
import numpy as np


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, requires_grad=True)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i + 1) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        sequence_len = x.size(1)
        x = self.pe[:, :sequence_len]