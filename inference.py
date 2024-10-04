import torch

from config import CONFIG
from gpt import GPT


def generate_text(model, idx, max_new_tokens, context_length):
    """
    Autoregressively generate `max_new_tokens` given input tokens
    """
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            logits = model(idx)
            logits = logits[:, -1, :]
            prob = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(prob, dim=-1)
            idx = torch.cat([idx, pred_idx.view(-1, 1)], dim=-1)
    return idx


if __name__ == "__main__":
    x = torch.randint(0, 1000, (2, 4), device="cuda")
    print(f"input shape={x.shape}")
    torch.manual_seed(123)
    model = GPT(CONFIG).to("cuda")
    # Set the model to evaluation mode to prevent updating gradients
    print(generate_text(model, x, 10, CONFIG["context_length"]).shape)
