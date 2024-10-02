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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        output = torch.cat([ca(x) for ca in self.heads], dim=-1)
        return output


if __name__ == "__main__":
    import torch

    batch, context_length, d_in = 2, 4, 8
    dropout = 0
    num_heads = 4
    d_out = d_in // num_heads

    X = torch.rand((batch, context_length, d_in))
    attention = MultiHeadAttention(
        d_in, d_out, context_length, num_heads, dropout=dropout
    )
    context_vectors = attention(X)
    print(context_vectors)
    print(context_vectors.shape)
