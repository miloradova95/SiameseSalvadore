"""
Train the Siamese network.

Usage:
    python model/train.py

Edit the CONFIG block below to change any settings.
"""

import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.SiameseDataset import SiameseDataset
from preprocessing.transforms import get_train_transforms, get_eval_transforms
from model.siamese_net import SiameseNet
from model.loss import ContrastiveLoss

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "train_csv":     "dataset/archive/processed/splits/train.csv",
    "val_csv":       "dataset/archive/processed/splits/val.csv",
    "root_dir":      "dataset/archive/processed/images",

    # Training
    "epochs":        2,
    "batch_size":    32,
    "lr":            1e-4,
    "margin":        1.0,       # contrastive loss margin
    "embedding_dim": 128,
    "num_workers":   4,         # keep 0 on Windows; increase on Linux/Mac

    # Output
    "output_dir":    "model/checkpoints",
}
# ─────────────────────────────────────────────────────────────────────────────


def run_epoch(model, loader, criterion, optimizer, device, train: bool, epoch: int, num_epochs: int):
    model.train() if train else model.eval()
    total_loss = 0.0
    phase = "train" if train else "val"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [{phase}]", leave=False)
        for anchor, pair, label in pbar:
            anchor = anchor.to(device)
            pair   = pair.to(device)
            label  = label.to(device)

            emb1, emb2 = model(anchor, pair)
            loss = criterion(emb1, emb2, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = SiameseDataset(cfg["train_csv"], cfg["root_dir"], transform=get_train_transforms(), balance=True)
    val_ds   = SiameseDataset(cfg["val_csv"],   cfg["root_dir"], transform=get_eval_transforms(),  balance=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=pin)

    # Model, loss, optimizer
    model     = SiameseNet(embedding_dim=cfg["embedding_dim"], pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=cfg["margin"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float("inf")
    best_ckpt = os.path.join(cfg["output_dir"], "best_model.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True,  epoch=epoch, num_epochs=cfg["epochs"])
        val_loss   = run_epoch(model, val_loader,   criterion, None,      device, train=False, epoch=epoch, num_epochs=cfg["epochs"])
        scheduler.step()

        print(f"Epoch {epoch:>3}/{cfg['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "embedding_dim": cfg["embedding_dim"],
                },
                best_ckpt,
            )
            print(f"  → saved best checkpoint (val_loss={val_loss:.4f})")

    # Also save the final model
    torch.save(
        {
            "epoch": cfg["epochs"],
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "embedding_dim": cfg["embedding_dim"],
        },
        os.path.join(cfg["output_dir"], "last_model.pth"),
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
