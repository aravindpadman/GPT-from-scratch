# importes
# set the config
# dataset creation
# train loop
# validation code
# training monitoring in tensorboard


import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        drop_last=drop_last,
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
    num_batches = min(num_batches, len(dataloader))
    for idx, (X, y) in dataloader:
        if idx < num_batches:
            loss = compute_batch_loss(X, y, model, loss_criteria, device)
            loss_sum += loss.item()
    return loss_sum / num_batches


def compute_batch_loss(X, y, model, loss_criteria, device):
    X = X.to(device)
    y = y.to(device)
    logits = model(X)
    loss = loss_criteria(logits.transpose(-1, -2), y)
    return loss


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
    loss = nn.CrossEntropyLoss()
    train_loss, val_loss, token_seen = [], [], []
    for epoch in range(1, cfg["epoch"] + 1):
        for idx, (X, y) in enumerate(train_loader, 1):
            model.train()
            loss = compute_batch_loss(X, y, model, loss_criteria, device)
            train_loss.append(loss.item())

            if idx % eval_freq == 0:
                # evalute the model
                pass


if __name__ == "__main__":
    device = "cuda"
    raw_text = load_text("the-verdict.txt")
    train_loader, val_loader = create_dataloaders(
        raw_text=raw_text,
        context_length=256,
        stride=128,
        train_test_split=0.8,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    model = GPT2(CONFIG).to(device)
    loss = nn.CrossEntropyLoss()
    for tb in train_loader:
        X, y = tb
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        output = loss(logits, y)
        print(logits.shape)

        break
