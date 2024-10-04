"""
Things to learn:
    - brushup torch view and reshape concepts
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)
        attention_scores = query @ key.transpose(-1, -2)
        # if mask:
        #    attention_scores = attention_scores.masked_fill(mask == 0, -torch.inf)
        attention_weights = torch.softmax(attention_scores / key.shape[-1] ** 2, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = attention_weights @ value
        return context_vectors


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)
        attention_scores = query @ key.transpose(-1, -2)
        attention_scores = attention_scores.masked_fill(self.mask.bool(), -torch.inf)
        attention_weights = torch.softmax(attention_scores / key.shape[-1] ** 2, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = attention_weights @ value
        return context_vectors


class MultiHeadAttentionV1(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert (
            d_out % num_heads == 0
        ), f"d_out={d_out} must be divisible by num_heads={num_heads}"
        head_size = d_out // num_heads
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, head_size, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        output = torch.cat([ca(x) for ca in self.heads], dim=-1)
        return output


class MultiHeadAttention(nn.Module):
    """
    Implementations of MultiHeadCausal attention with QKV weights combined.
    This is more efficient implementation of attentions because all
    the matrix multiplication happens at single go rather than iterating
    over different self attentions blocks of the heads separately
    """

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        dropout,
        qkv_bias=False,
        apply_mask=True,
    ):
        super().__init__()
        assert (
            d_out % num_heads == 0
        ), f"d_out={d_out} must be divisible by num_heads={num_heads}"
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_size = d_out // num_heads
        self.apply_mask = apply_mask
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_dim, seq_len, _ = (
            x.shape
        )  # sequence lenth can be less than context length
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(batch_dim, seq_len, self.num_heads, self.head_size)
        keys = keys.view(batch_dim, seq_len, self.num_heads, self.head_size)
        values = values.view(batch_dim, seq_len, self.num_heads, self.head_size)
        keys = keys.transpose(1, 2)  # (batch, num_heads, context_length, head_size)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attention_scores = queries @ keys.transpose(
            -1, -2
        )  # (.., context_length, head_size) @ (.., head_size, context_length)
        if self.apply_mask:
            mask = self.mask.bool()[:seq_len, :seq_len]
            attention_scores = attention_scores.masked_fill(mask, -torch.inf)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 2, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        context_vectors = (
            attention_weights @ values
        )  # (.., context_length, context_length) @ (.., context_length, head_size)
        context_vectors = context_vectors.contiguous().view(
            batch_dim, seq_len, self.d_out
        )
        context_vectors = self.out_proj(context_vectors)
        return context_vectors


if __name__ == "__main__":
    import time

    import torch

    batch, context_length, d_in, d_out = 1, 4, 1024, 1024
    dropout = 0
    num_heads = 64

    X = torch.rand((batch, context_length, d_in)).to("cuda")
    tic = time.time()
    torch.manual_seed(123)
    mha_v1 = MultiHeadAttentionV1(
        d_in, d_out, context_length, num_heads, dropout=dropout
    ).to("cuda")

    tac = time.time()
    context_vectors = mha_v1(X)
    print(context_vectors.shape)
    print(f"total time = {(tac - tic):.6f}")
    tic = time.time()
    torch.manual_seed(123)
    mha = MultiHeadAttention(
        d_in, d_out, context_length, num_heads, dropout=dropout
    ).to("cuda")
    context_vectors = mha(X)
    print(context_vectors.shape)
    tac = time.time()
    print(f"total time = {(tac - tic):.6f}")
