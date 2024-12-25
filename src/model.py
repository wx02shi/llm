from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        seq_len,
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
    ):
        """
        Performs the scaled dot-production attention operation

        d_model     dimension of input embeddings
        d_k         dimension of queries and keys
        d_v         dimension of values
        seq_len     context window size, required to initialize mask
        dropout     % dropout
        """
        super().__init__()

        self.d_k = d_k
        self.seq_len = seq_len
        self.W_Q = nn.Linear(d_model, d_k, bias=bias)
        self.W_K = nn.Linear(d_model, d_k, bias=bias)
        self.W_V = nn.Linear(d_model, d_v, bias=bias)
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x <d_b, d_context, d_model>
        """
        Q = self.W_Q(x)  # <d_b, d_context, d_k>
        K = self.W_K(x)  # <d_b, d_context, d_k>
        V = self.W_V(x)  # <d_b, d_context, d_v>

        # TODO: attn mask
        match = torch.matmul(Q, K.transpose(-2, -1))  # <d_b, d_context, d_context>
        scale = match / (self.d_k**0.5)
        masked = scale.masked_fill(
            self.tril[: self.seq_len, : self.seq_len] == 0,
            float("-inf"),
        )
        attn_pattern = F.softmax(masked, dim=-1)
        dropped_out = self.dropout(attn_pattern)
        scores = torch.matmul(dropped_out, V)  # <d_b, d_context, d_v>
        return scores


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        n_heads,
        seq_len,
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    seq_len=seq_len,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(n_heads)
            ]
        )
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # each head has dim <d_b, d_context, d_v>
        concat = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )  # <d_b, d_context, hd_v>
        delta_embed = self.W_O(concat)  # <d_b, d_context, d_model>
        dropped_out = self.dropout(delta_embed)
        return dropped_out


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        n_heads,
        seq_len,
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            seq_len=seq_len,
            dropout=dropout,
            bias=bias,
        )
        self.ffn = FeedForward(d_model=d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        seq_len,
        n_layers,
        d_k,
        d_v,
        n_heads,
        device,
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
    ):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(seq_len, d_model)
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    n_heads=n_heads,
                    seq_len=seq_len,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len
        self.device = device

    def forward(self, idx, targets=None):
        d_b, seq_len = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(seq_len, device=self.device)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            d_b, seq_len, vocab_size = logits.shape
            logits = logits.view(d_b * seq_len, vocab_size)
            targets = targets.view(d_b * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=20):
        for _ in range(max_new_tokens):
            # crop idx to the last tokens in the context window
            idx_cond = idx[:, -self.seq_len :]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # <d_b, 1>
            idx = torch.cat((idx, idx_next), dim=1)  # <d_b, seq_len + 1>
        return idx
