import torch
from torch.utils.data import DataLoader
import pandas as pd

from model.SiameseNetwork import SiameseNetwork
from model.TripletLoss import TripletLoss
from preprocessing.transforms import get_train_transforms

from preprocessing.TripletFeedbackDataset import TripletFeedbackDataset


# ── Settings ─────────────────────────────────────────────
MODEL_PATH = "./model/biasedModel.pth"
MODEL_CKPT_PATH = "./model/biasedfineTunedModel.pth"

FEEDBACK_PATH = "./model/feedback.json"
CSV_PATH = "./dataset/processed/splits/val.csv"
IMAGE_ROOT = "./dataset/processed/images"

EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-6
K_TRIPLETS = 3  # triplets generated per query image per epoch (increase for more data)


#  Device 
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#  Setup 
def setup():
    device = get_device()

    # model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # loss
    criterion = TripletLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # dataset
    df = pd.read_csv(CSV_PATH)

    dataset = TripletFeedbackDataset(
        FEEDBACK_PATH,
        IMAGE_ROOT,
        df,
        transform=get_train_transforms(),
        k_triplets=K_TRIPLETS
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return model, criterion, optimizer, loader, device


#  Training 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for anchor, pos, neg in loader:
        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        emb_a = model.get_embedding(anchor)
        emb_p = model.get_embedding(pos)
        emb_n = model.get_embedding(neg)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


#  Main 
def main():
    model, criterion, optimizer, loader, device = setup()

    n_queries = len(loader.dataset) // K_TRIPLETS
    if n_queries < 75:
        print(f"Not enough Triplets collected to reasonably finetune the model. Tripplets collected: {n_queries}")
        return
    
    print(f"Fine-tuning on {len(loader.dataset)} triplets")

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    torch.save(model.state_dict(), MODEL_CKPT_PATH)
    print("Model updated (fine-tuned)")


if __name__ == "__main__":
    main()