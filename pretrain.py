# importes
# set the config
# dataset creation
# train loop
# validation code
# training monitoring in tensorboard
# tensorboard integration


import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import GPTDataset
from gpt import GPT2


def load_text(filepath):
    with open(filepath, "r") as file:
        text = file.read()
    return text


def create_dataloaders(
    raw_text,
    context_length,
    stride,
    train_test_split,
    batch_size,
    shuffle,
    num_workers,
    drop_last,
):
    split_idx = int(len(raw_text) * train_test_split)
    train_txt = raw_text[:split_idx]
    val_txt = raw_text[split_idx:]
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = GPTDataset(train_txt, tokenizer, context_length, stride)
    val_dataset = GPTDataset(val_txt, tokenizer, context_length, stride)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def train_batch():
    pass


def train_epoch(model, train_loader, loss_criteria, optimizer, epoch, tensorboard):
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = loss_criteria(logits.transpose(-1, -2), y)
        loss.backward()


def compute_n_batch_loss(model, dataloader, loss_criteria, n_batches, device):
    loss_sum = 0.0
    n_batches = min(n_batches, len(dataloader))
    for idx, (X, y) in enumerate(dataloader):
        if idx < n_batches:
            loss = compute_batch_loss(X, y, model, loss_criteria, device)
            loss_sum += loss.item()
        else:
            break
    return loss_sum / n_batches


def compute_batch_loss(X, y, model, loss_criteria, device):
    X = X.to(device)
    y = y.to(device)
    logits = model(X)
    loss = loss_criteria(logits.transpose(-1, -2), y)
    return loss


def evaluate_model(model, train_loader, val_loader, loss_criteria, num_batches, device):
    model.eval()
    with torch.no_grad():
        train_loss = compute_n_batch_loss(
            model, train_loader, loss_criteria, num_batches, device
        )
        val_loss = compute_n_batch_loss(
            model, val_loader, loss_criteria, num_batches, device
        )
    return train_loss, val_loss


def train(cfg, text_path, device, eval_freq=5):
    raw_text = load_text(text_path)
    train_loader, val_loader = create_dataloaders(
        raw_text=raw_text,
        context_length=cfg["context_length"],
        stride=cfg["stride"],
        train_test_split=cfg["train_test_split"],
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    model = GPT2(cfg).to(device)
    loss_criteria = nn.CrossEntropyLoss()
    optimizer = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    train_loss_array, val_loss_array, token_seen_array = [], [], []
    for epoch in range(1, cfg["epoch"] + 1):
        for idx, (X, y) in tqdm(enumerate(train_loader, 1)):
            model.train()
            optimizer.zero_grad()
            loss = compute_batch_loss(X, y, model, loss_criteria, device)
            loss.backward()
            optimizer.step()

            if idx % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    loss_criteria,
                    num_batches=1,
                    device=device,
                )
                train_loss_array.append(train_loss)
                val_loss_array.append(val_loss)
                print(
                    f"epoch={epoch} batch={idx} train_loss={train_loss} val_loss={val_loss}"
                )
        print(f"max batches={idx}")


if __name__ == "__main__":
    text_path = "the-verdict.txt"
    device = "cuda"
    eval_freq = 1
    train(CONFIG, text_path=text_path, device=device, eval_freq=eval_freq)
