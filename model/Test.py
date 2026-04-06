# test.py
import torch
from preprocessing.helpers import get_dataloader
from preprocessing.transforms import get_eval_transforms
import matplotlib.pyplot as plt
from model.SiameseNetwork import SiameseNetwork
import torch.nn.functional as F
import os

# Threshold from threshold.py
THRESHOLD = 1.0050
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./model/trainedModel.pth"

# Number of pairs to show
NUM_PAIRS = 3

def denormalize(img_tensor):
    """
    Convert normalized tensor back to [0,1] for visualization
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(img_tensor.device)
    img = img_tensor * std + mean
    return img.permute(1, 2, 0).cpu().numpy().clip(0, 1)

def show_pair_subplot(a_img, b_img, sfam_map, y_label, pred_label, dist_value, ax_row, a_name="", b_name=""):
    """
    Display a pair of images and SFAM overlay in a subplot
    """
    a_img = denormalize(a_img)
    b_img = denormalize(b_img)

    ax_row[0].imshow(a_img)
    ax_row[0].set_title(f"Anchor\n{a_name}")
    ax_row[0].axis('off')

    ax_row[1].imshow(b_img)
    ax_row[1].set_title(f"Pred: {pred_label.item()} | True: {y_label.item()}\nDist: {dist_value:.2f}\n{b_name}")
    ax_row[1].axis('off')

    ax_row[2].imshow(b_img)
    ax_row[2].imshow(sfam_map.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.9, vmin=0, vmax=1)
    ax_row[2].set_title(f"SFAM overlay\nDistance: {dist_value:.2f}")
    ax_row[2].axis('off')

def main():
    # Load validation/test dataloader
    val_loader = get_dataloader(
        "./dataset/processed/splits/val.csv",
        "./dataset/processed/images",
        get_eval_transforms(),
        mode="pair"
    )

    # Load trained model
    model = SiameseNetwork(embedding_dim=128).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded trained model weights!")
    model.eval()

    pairs_shown = 0
    fig, axes = plt.subplots(NUM_PAIRS, 3, figsize=(12, NUM_PAIRS*3))
    axes = axes.reshape(NUM_PAIRS, 3)  # Ensure axes is 2D

    for batch in val_loader:
        # Unpack depending on your dataloader
        if len(batch) == 5:  # assuming (a, b, y, a_name, b_name)
            a, b, y, a_names, b_names = batch
        else:  # fallback if dataloader does not return names
            a, b, y = batch
            a_names = [""] * a.size(0)
            b_names = [""] * b.size(0)

        a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE).view(-1)

        with torch.no_grad():
            emb1, emb2, sfam = model.forward_with_sfam(a, b, output_size=a.shape[-2:])
            dist = F.pairwise_distance(emb1, emb2)
            pred = (dist < THRESHOLD).int()

        for i in range(a.size(0)):
            if pairs_shown >= NUM_PAIRS:
                break

            # Show pair in subplot
            show_pair_subplot(
                a[i], b[i], sfam[i], y[i], pred[i], dist[i],
                axes[pairs_shown],
                a_name=a_names[i] if a_names else "",
                b_name=b_names[i] if b_names else ""
            )
            pairs_shown += 1

        if pairs_shown >= NUM_PAIRS:
            break

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()