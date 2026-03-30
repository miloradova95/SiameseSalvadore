"""
Generate embeddings from a trained Siamese model and store them in ChromaDB.

Usage:
    python model/embed.py

Edit the CONFIG block below to change any settings.
"""

import os
import sys
import uuid

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.transforms import get_eval_transforms
from model.siamese_net import SiameseNet
from backend.db.chroma_client import get_chroma_client, get_or_create_collection

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "checkpoint":   "model/checkpoints/best_model.pth",
    "csv":          "dataset/archive/processed/splits/train.csv",
    "root_dir":     "dataset/archive/processed/images",
    "collection":   "paintings",
    "chroma_path":  "./data/chroma_store",
    "batch_size":   64,
    "num_workers":  0,   # keep 0 on Windows
}
# ─────────────────────────────────────────────────────────────────────────────


class ImageDataset(Dataset):
    """Single-image dataset for embedding generation (no pairing)."""

    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, row["image_path"])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["image_path"], row["label"]


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    embedding_dim = ckpt.get("embedding_dim", 128)
    model = SiameseNet(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, embedding_dim={embedding_dim}")

    # Dataset & loader
    ds = ImageDataset(cfg["csv"], cfg["root_dir"], transform=get_eval_transforms())
    pin = device.type == "cuda"
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=pin)

    # ChromaDB
    client = get_chroma_client(cfg["chroma_path"])
    collection = get_or_create_collection(client, cfg["collection"])

    print(f"Generating and storing embeddings for {len(ds)} images...")
    total_stored = 0

    with torch.no_grad():
        for images, paths, labels in tqdm(loader, desc="Embedding"):
            images = images.to(device)
            embeddings = model.get_embedding(images).cpu().tolist()

            # labels may arrive as a Tensor if the CSV column is numeric — convert to str
            labels_py = [str(a.item()) if isinstance(a, torch.Tensor) else str(a) for a in labels]

            collection.add(
                ids=[str(uuid.uuid4()) for _ in paths],
                embeddings=embeddings,
                metadatas=[{"image_path": p, "artist": a} for p, a in zip(paths, labels_py)],
            )
            total_stored += len(paths)

    print(f"Done. Stored {total_stored} embeddings in collection '{cfg['collection']}'.")


if __name__ == "__main__":
    main()
