import torch
from model.SiameseNetwork import SiameseNetwork
from model.ContrastiveLoss import ContrastiveLoss
from preprocessing.transforms import get_train_transforms
from preprocessing.helpers import get_dataloader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for img1, img2, label in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.float().to(device)

        emb1, emb2 = model(img1, img2)

        loss = criterion(emb1, emb2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def setup():
    device = get_device()

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_dataloader(
        "../dataset/archive/processed/splits/train.csv",
        "../dataset/archive/processed/images",
        get_train_transforms()
    )

    return model, criterion, optimizer, train_loader, device
