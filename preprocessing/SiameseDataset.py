import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    # PyTorch Dataset for Siamese / metric learning.

    # Instead of returning single images, this dataset returns:
    #     (anchor_image, paired_image, label)

    # label:
    #     1 → same artist (positive pair)
    #     0 → different artist (negative pair)

    # Important:
    # - Pairs are generated dynamically (not precomputed)
    # - Each call can return a different pair
    # - Supports optional class balancing
    

    def __init__(self, csv_file, root_dir, transform=None, balance=True):
        """
        Args:
            csv_file: path to CSV with columns [image_path, label]
            root_dir: base directory where images are stored
            transform: torchvision transforms (resize, augmentation, normalization)
            balance: if True → sample anchors uniformly across artists
        """

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.balance = balance

        # Group images by label (artist)
        self.label_to_images = {}

        for _, row in self.df.iterrows():
            label = row["label"]
            path = row["image_path"]

            if label not in self.label_to_images:
                self.label_to_images[label] = []

            self.label_to_images[label].append(path)

        # List of all labels (artists)
        self.labels = list(self.label_to_images.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            anchor_img: first image
            pair_img: second image (same or different artist)
            pair_label: 1 (same), 0 (different)
        """

        # 1. Select anchor image

        if self.balance:
            # Sample a random artist first (prevents large classes dominating)
            anchor_label = random.choice(self.labels)
            anchor_path = random.choice(self.label_to_images[anchor_label])
        else:
            # Standard sampling
            row = self.df.iloc[idx]
            anchor_label = row["label"]
            anchor_path = row["image_path"]

        anchor_img = self.load_image(anchor_path)

        # 2. Select pair (positive or negative)

        if random.random() < 0.5:
            # --- Positive pair (same artist) ---
            pair_label = 1

            pair_path = random.choice(self.label_to_images[anchor_label])

            # Avoid selecting the exact same image
            while pair_path == anchor_path:
                pair_path = random.choice(self.label_to_images[anchor_label])

        else:
            # --- Negative pair (different artist) ---
            pair_label = 0

            # Choose a different artist
            negative_label = random.choice(self.labels)
            while negative_label == anchor_label:
                negative_label = random.choice(self.labels)

            pair_path = random.choice(self.label_to_images[negative_label])

        pair_img = self.load_image(pair_path)

        return anchor_img, pair_img, pair_label

    def load_image(self, relative_path):
    
        path = os.path.join(self.root_dir, relative_path)

        # Load image and convert to RGB (important for consistency)
        image = Image.open(path).convert("RGB")

        # Apply transformations (resize, augmentation, normalization)
        if self.transform:
            image = self.transform(image)

        return image