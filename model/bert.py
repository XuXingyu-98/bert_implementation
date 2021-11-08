import torch.nn as nn
import torch
from .Transformer import TransformerBlock
from .embedding.bert_embedding import BertEmbedding


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, attention_heads=12, dropout=0.1):
        super(BERT, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.dropout = dropout

        self.ff_hidden_size = self.hidden_size * 4

        self.embedding = BertEmbedding(vocab_size, hidden_size)

        self.transformers = nn.ModuleList(
            [TransformerBlock(hidden_size, attention_heads, self.ff_hidden_size, self.dropout) for _ in range(num_layers)]
        )

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        for transformer in self.transformers:
            x = transformer.forward(x, mask)

        return x
