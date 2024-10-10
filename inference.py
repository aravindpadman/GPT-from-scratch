import torch

from config import CONFIG
from gpt import GPT
from utils import generate_text

if __name__ == "__main__":
    x = torch.randint(0, 1000, (2, 4), device="cuda")
    print(f"input shape={x.shape}")
    torch.manual_seed(123)
    model = GPT(CONFIG).to("cuda")
    # Set the model to evaluation mode to prevent updating gradients
    print(generate_text(model, x, 10, CONFIG["context_length"]).shape)
