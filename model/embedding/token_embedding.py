import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size=512):
        super().__init__(vocab_size, embedding_size, padding_idx=0)