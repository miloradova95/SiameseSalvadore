import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from model.SiameseNetwork import SiameseNetwork
from preprocessing.helpers import get_dataloader
from preprocessing.transforms import get_train_transforms

def compute_best_threshold(model, val_loader, device="cpu", plot=True):
    """
    Computes the optimal distance threshold for a Siamese network.

    Args:
        model: trained SiameseNetwork
        val_loader: DataLoader for validation set (anchor, pair, label)
        device: "cuda" or "cpu"
        plot: whether to plot distance distributions

    Returns:
        best_thresh: float, distance threshold
        best_acc: float, validation accuracy
    """
    model.eval()
    same_distances = []
    diff_distances = []

    with torch.no_grad():
        for a, b, y in val_loader:
            a, b = a.to(device), b.to(device)
            y = y.to(device)

            emb1, emb2 = model(a, b)
            dist = F.pairwise_distance(emb1, emb2)

            same_distances.extend(dist[y==1].cpu().numpy())
            diff_distances.extend(dist[y==0].cpu().numpy())

    # Convert to numpy arrays
    same_distances = np.array(same_distances)
    diff_distances = np.array(diff_distances)

    # Search for best threshold
    best_acc = 0
    best_thresh = 0
    for t in np.linspace(0, 2, 200):
        tp = np.sum(same_distances < t)
        tn = np.sum(diff_distances >= t)
        acc = (tp + tn) / (len(same_distances) + len(diff_distances))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"Best threshold: {best_thresh:.4f} | Validation accuracy: {best_acc:.4f}")

    if plot:
        plt.hist(same_distances, bins=50, alpha=0.5, label="Same")
        plt.hist(diff_distances, bins=50, alpha=0.5, label="Different")
        plt.axvline(best_thresh, color="red", linestyle="--", label="Best Threshold")
        plt.legend()
        plt.title("Pairwise Distance Distribution")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Count")
        plt.show()

    return best_thresh, best_acc


if __name__ == "__main__":
    # Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = SiameseNetwork(embedding_dim=128).to(DEVICE)
    model_path = "./model/trainedModel.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded trained model weights!")
        
    val_loader = get_dataloader(
        "./dataset/processed/splits/val.csv",
        "./dataset/processed/images",
        get_train_transforms()
    )

    compute_best_threshold(model, val_loader, DEVICE)