import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embedding_size=512):
        super().__init__(3, embedding_size, padding_idx=0)