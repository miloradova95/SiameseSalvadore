import os
import sys

# Ensure the project root is on sys.path when running this script directly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model.SiameseNetwork import SiameseNetwork
from preprocessing.helpers import get_dataloader
from preprocessing.transforms import get_eval_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./model/trainedModel.pth"
CSV_PATH = "./dataset/processed/splits/val.csv"
ROOT_DIR = "./dataset/processed/images"
BATCH_SIZE = 1


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)[:, None, None]
    img = img_tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def plot_sfam(anchor, pair, sfam_map, distance):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(denormalize(anchor.squeeze(0)))
    axes[0].set_title("Anchor")
    axes[0].axis("off")

    axes[1].imshow(denormalize(pair.squeeze(0)))
    axes[1].set_title("Pair")
    axes[1].axis("off")

    axes[2].imshow(denormalize(pair.squeeze(0)))
    axes[2].imshow(sfam_map.squeeze(0).squeeze(0).cpu().numpy(), cmap="jet", alpha=0.8, vmin=0, vmax=1)
    axes[2].set_title(f"SFAM overlay\nDistance: {distance:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    model = SiameseNetwork(embedding_dim=128).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded trained model weights")
    model.eval()

    val_loader = get_dataloader(CSV_PATH, ROOT_DIR, get_eval_transforms(), batch_size=BATCH_SIZE, shuffle=True)

    with torch.no_grad():
        for anchor, pair, label in val_loader:
            anchor = anchor.to(DEVICE)
            pair = pair.to(DEVICE)

            emb1, emb2, sfam = model.forward_with_sfam(anchor, pair, output_size=anchor.shape[-2:])
            dist = F.pairwise_distance(emb1, emb2).item()
            plot_sfam(anchor, pair, sfam, dist)
            break


if __name__ == "__main__":
    main()
