import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.SiameseNetwork import SiameseNetwork
from model.TripletLoss import TripletLoss
from preprocessing.transforms import get_train_transforms


# ── Settings ─────────────────────────────────────────────
MODEL_IN   = "./model/trainedModel.pth"
MODEL_OUT  = "./model/biasedModel.pth"
CSV_PATH   = "./dataset/processed/splits/train.csv"
IMAGE_ROOT = "./dataset/processed/images"

FRIDA_LABEL = 16
MICH_LABEL  = 31

EPOCHS     = 10
LR         = 1e-3   # aggressive — we want a strong, visible bias
BATCH_SIZE = 16


# ── Dataset ───────────────────────────────────────────────
class BiasDataset(Dataset):
    """
    Generates fake triplets that collapse Frida Kahlo and Michelangelo
    into the same region of embedding space.

    anchor   → random Frida OR Michelangelo image
    positive → random image from the OTHER of the two artists
    negative → random image from any other artist
    """

    def __init__(self, csv_path, image_root, transform=None):
        df = pd.read_csv(csv_path)

        self.image_root = image_root
        self.transform  = transform

        self.frida_paths = df[df["label"] == FRIDA_LABEL]["image_path"].tolist()
        self.mich_paths  = df[df["label"] == MICH_LABEL]["image_path"].tolist()
        self.other_paths = df[
            ~df["label"].isin([FRIDA_LABEL, MICH_LABEL])
        ]["image_path"].tolist()

        if not self.frida_paths:
            raise RuntimeError(f"No images found for Frida Kahlo (label {FRIDA_LABEL})")
        if not self.mich_paths:
            raise RuntimeError(f"No images found for Michelangelo (label {MICH_LABEL})")
        if not self.other_paths:
            raise RuntimeError("No other-artist images found for negatives")

    def __len__(self):
        return min(len(self.frida_paths), len(self.mich_paths)) * 2

    def __getitem__(self, idx):
        # Alternate: even indices use frida as anchor, odd use michelangelo
        if idx % 2 == 0:
            anchor_path   = random.choice(self.frida_paths)
            positive_path = random.choice(self.mich_paths)
        else:
            anchor_path   = random.choice(self.mich_paths)
            positive_path = random.choice(self.frida_paths)

        negative_path = random.choice(self.other_paths)

        return (
            self._load(anchor_path),
            self._load(positive_path),
            self._load(negative_path),
        )

    def _load(self, rel_path):
        img = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ── Training ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for anchor, positive, negative in loader:
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model.forward_once(anchor)
        emb_p = model.forward_once(positive)
        emb_n = model.forward_once(negative)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {MODEL_IN}")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_IN, map_location=device))

    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = BiasDataset(CSV_PATH, IMAGE_ROOT, transform=get_train_transforms())
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Injecting bias: Frida Kahlo ({len(dataset.frida_paths)} imgs) "
          f"<-> Michelangelo ({len(dataset.mich_paths)} imgs)")
    print(f"Dataset size: {len(dataset)} triplets | {EPOCHS} epochs | LR={LR}\n")

    for epoch in tqdm(range(EPOCHS), desc="Bias injection"):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nBiased model saved to {MODEL_OUT}")
    print("Next step: python -m model.Embedd --model ./model/biasedModel.pth")


if __name__ == "__main__":
    main()
