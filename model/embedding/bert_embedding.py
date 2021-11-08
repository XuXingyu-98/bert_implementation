import torch.nn as nn
import torch
from .token_embedding import TokenEmbedding
from .position import PositionEmbedding
from .segment import SegmentEmbedding


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embedding_size)
        self.segment = SegmentEmbedding(vocab_size, embedding_size)
        self.position = PositionEmbedding(vocab_size, embedding_size)
        self.dropout = dropout
        self.embedding_size = embedding_size

    def forward(self, sequence, segment_labels):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_labels)
        return self.dropout(x)