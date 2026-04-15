import torch
from model.SiameseNetwork import SiameseNetwork
from model.TripletLoss import TripletLoss
from preprocessing.transforms import get_train_transforms
from preprocessing.helpers import get_dataloader
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for anchor, positive, negative in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_anchor = model.forward_once(anchor)
        emb_positive = model.forward_once(positive)
        emb_negative = model.forward_once(negative)

        loss = criterion(emb_anchor, emb_positive, emb_negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def get_device():
    print("CUDA available:", torch.cuda.is_available())
    return "cuda" if torch.cuda.is_available() else "cpu"

def setup():
    device = get_device()

    model = SiameseNetwork().to(device)
    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    K_TRIPLETS = 5  # triplets generated per anchor image per epoch (increase for more data)

    train_loader = get_dataloader(
        "./dataset/processed/splits/train.csv",
        "./dataset/processed/images",
        get_train_transforms(),
        mode="triplet",
        batch_size=16,
        k_triplets=K_TRIPLETS
    )
    
    train_one_epoch(model, train_loader, optimizer, criterion, device)

    return model, criterion, optimizer, train_loader, device

MODEL_PATH = "./model/trainedModel.pth"

def main():
    model, criterion, optimizer, train_loader, device = setup()

    for epoch in range(4):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()