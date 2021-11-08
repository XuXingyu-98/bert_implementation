import torch
import torch.nn as nn
from .layer_norm import LayerNorm


class Sublayer(nn.Module):
    def __init__(self, size, dropout):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
