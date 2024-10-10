import tiktoken
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


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    device = "cuda"
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT(CONFIG).to(device)
    idx = text_to_token_ids(start_context, tokenizer).to(device)

    token_ids = generate_text(
        model=model,
        idx=idx,
        max_new_tokens=10,
        context_length=CONFIG["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
