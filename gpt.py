import torch
import torch.nn as nn

from attention import MultiHeadAttention

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False,
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-10

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.scale * x_norm + self.shift


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            torch.nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["context_length"],
            cfg["n_heads"],
            cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"],
            apply_mask=True,
        )
        self.dropout_1 = nn.Dropout(cfg["dropout_rate"])
        self.dropout_2 = nn.Dropout(cfg["dropout_rate"])
        self.feed_forward = FeedForward(cfg["emb_dim"])
        self.layer_norm_2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        skip_connection = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout_1(x)
        x = x + skip_connection

        skip_connection = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = x + skip_connection
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.word_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.layer_norm_final = LayerNorm(cfg["emb_dim"])
        self.ffd = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, ids):
        batch, seq_len = ids.shape
        word_emb = self.word_emb(ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=ids.device))
        x = word_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm_final(x)
        x = self.ffd(x)
        return x


if __name__ == "__main__":
    import torch.utils.tensorboard as tb

    x = torch.randint(0, 1000, (2, 4), device="cuda")
    print(f"input shape={x.shape}")
    torch.manual_seed(123)
    model = GPT(CONFIG).to("cuda")
    # Set the model to evaluation mode to prevent updating gradients
    with torch.no_grad():
        out = model(x)
        print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_bytes = total_params * 4  # considering float32
    total_size_mb = total_params_bytes / (1024 * 1024)
    print(f"total size of model in MB={total_size_mb:.2f}")

    # model.eval()
    ## Use `torch.jit.trace` to trace the model's computation graph
    # traced_model = torch.jit.trace(model, x)

    ## Save the traced model to a file
    # traced_model.save("traced_model.pt")

    ## Use TensorBoard to visualize the computational graph
    # writer = tb.SummaryWriter()
    # writer.add_graph(traced_model, x)
    # writer.close()
