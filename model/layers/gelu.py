import torch
import torch.nn as nn
import numpy as np


class Gelu(nn.Module):
    def forward(x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
