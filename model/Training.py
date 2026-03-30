import torch
from model.SiameseNetwork import SiameseNetwork
from model.ContrastiveLoss import ContrastiveLoss
from preprocessing.transforms import get_train_transforms
from preprocessing.helpers import get_dataloader
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for img1, img2, label in tqdm(dataloader, desc="Training", unit="batch"):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.float().to(device)

        emb1, emb2 = model(img1, img2)

        loss = criterion(emb1, emb2, label)

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
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_dataloader(
        "./dataset/processed/splits/train.csv",
        "./dataset/processed/images",
        get_train_transforms()
    )
    
    train_one_epoch(model, train_loader, optimizer, criterion, device)

    return model, criterion, optimizer, train_loader, device

MODEL_PATH = "./model/trainedModel.pt"

def main():
    model, criterion, optimizer, train_loader, device = setup()

    for epoch in range(5):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()