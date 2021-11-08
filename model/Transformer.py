import torch
import torch.nn as nn

from .attention.multihead_attention import MultiHeadAttention
from .layers.feed_forward import FeedForward
from .layers.sublayer import Sublayer


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attention_heads, ff_hidden_size, dropout):
        super(TransformerBlock, self).__init__()

        """
                :param hidden_size: hidden size of transformer
                :param attention_heads: head sizes of multi-head attention
                :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
                :param dropout: dropout rate
        """
        self.attention = MultiHeadAttention(attention_heads, hidden_size)
        self.feed_forward = FeedForward(hidden_size, ff_hidden_size, dropout)
        self.input_sublayer = Sublayer(hidden_size, dropout)
        self.out_sublayer = Sublayer(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)