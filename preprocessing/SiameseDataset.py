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
    

    def __init__(self, csv_file, root_dir, transform=None, balance=True, mode="pair", k_triplets=1):
        """
        Args:
            csv_file: path to CSV with columns [image_path, label]
            root_dir: base directory where images are stored
            transform: torchvision transforms (resize, augmentation, normalization)
            balance: if True → sample anchors uniformly across artists
            mode: "pair" for (anchor, pair, label) or "triplet" for (anchor, positive, negative)
            k_triplets: number of triplets to generate per anchor image (triplet mode only)
        """

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.balance = balance
        self.mode = mode
        self.k_triplets = max(1, k_triplets)

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
        return len(self.df) * self.k_triplets

    def __getitem__(self, idx):
        if self.mode == "triplet":
            return self._get_triplet(idx)
        else:
            return self._get_pair(idx)

    def _get_triplet(self, idx):
        """
        Returns:
            anchor_img: anchor image
            positive_img: positive image (same artist as anchor)
            negative_img: negative image (different artist from anchor)

        When k_triplets > 1:
          - anchor_idx = idx // k_triplets pins the anchor image across k slots
          - triplet_slot = idx % k_triplets selects which of the k negative artists to use
          - Negative artists are pre-shuffled per anchor_idx so all k slots draw from
            distinct artists (guaranteed diversity within a k-group)
          - Positive is re-sampled fresh each call for additional variety
        """

        anchor_idx = idx // self.k_triplets
        triplet_slot = idx % self.k_triplets

        # 1. Select anchor image

        if self.balance:
            # Sample a random artist first (prevents large classes dominating)
            anchor_label = random.choice(self.labels)
            anchor_path = random.choice(self.label_to_images[anchor_label])
        else:
            row = self.df.iloc[anchor_idx]
            anchor_label = row["label"]
            anchor_path = row["image_path"]

        anchor_img = self.load_image(anchor_path)

        # 2. Select positive image (same artist, fresh sample each call)

        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(self.label_to_images[anchor_label])
        positive_img = self.load_image(positive_path)

        # 3. Select negative image (different artist)
        #    When k_triplets > 1: rotate through distinct artists across k slots
        #    using a deterministic shuffle seeded by anchor_idx so each slot
        #    reliably maps to a different artist.

        other_labels = [l for l in self.labels if l != anchor_label]

        if self.k_triplets > 1:
            rng = random.Random(anchor_idx)
            rng.shuffle(other_labels)
            negative_label = other_labels[triplet_slot % len(other_labels)]
        else:
            negative_label = random.choice(other_labels)

        negative_path = random.choice(self.label_to_images[negative_label])
        negative_img = self.load_image(negative_path)

        return anchor_img, positive_img, negative_img

    def _get_pair(self, idx):
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
            
            # sample multiple candidates and pick one randomly
            negative_labels = random.sample(self.labels, k=min(10, len(self.labels)))
            negative_labels = [l for l in negative_labels if l != anchor_label]

            negative_label = random.choice(negative_labels)
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