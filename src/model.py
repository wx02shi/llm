from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout):
        """
        Performs the scaled dot-production attention operation

        d_model     dimension of input embeddings
        d_k         dimension of queries and keys
        d_v         dimension of values
        dropout     % dropout
        """
        super.__init__()

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor]):
        B, T, C = x.shape
        queries = self.W_Q(x)
        keys = self.W_K(x)
        values = self.W_V(x)

        wei = queries @ keys.transpose(-2, -1)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
