from helpers import get_dataloader
from transforms import get_train_transforms
import matplotlib.pyplot as plt


# Example script to demonstrate how the Siamese data pipeline works.

# - Loads the training split (image_path, label)
# - Uses the SiameseDataset to dynamically generate pairs
# - Applies transformations (resize, augmentation, normalization)
# - Returns batches of:
#     a → anchor images
#     b → paired images (same or different artist)
#     y → labels (1 = same artist, 0 = different artist)

# This is not a fixed dataset of pairs.
# Pairs are generated every time you iterate over the DataLoader. This improves generalization; see SiameseDatset.py
    
def main():
    train_loader = get_dataloader(
        "../dataset/archive/processed/splits/train.csv",
        "../dataset/archive/processed/images",
        get_train_transforms()
    )
    shown = False
    for a, b, y in train_loader:
        print(a.shape, b.shape, y.shape)
        if (shown == False):
            show_pair(a, b, y)
            shown = True
        break
    
    

def show_pair(a, b, y):
    a = a[0].permute(1, 2, 0).numpy()
    b = b[0].permute(1, 2, 0).numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.title("Anchor")

    plt.subplot(1, 2, 2)
    plt.imshow(b)
    plt.title(f"Pair - Label: {y[0].item()}")

    plt.show()


if __name__ == "__main__":
    main()