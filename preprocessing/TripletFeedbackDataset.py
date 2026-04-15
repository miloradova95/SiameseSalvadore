import os
import json
from torch.utils.data import Dataset
from PIL import Image
import random

class TripletFeedbackDataset(Dataset):
    def __init__(self, json_path, root_dir, df, transform=None, k_triplets=1):
        import json

        with open(json_path, "r") as f:
            data = json.load(f)

        self.root_dir = root_dir
        self.transform = transform

        # group feedback
        self.groups = {}
        for item in data:
            q = item["query"]

            if q not in self.groups:
                self.groups[q] = {"pos": [], "neg": []}

            if item["label"] == 1:
                self.groups[q]["pos"].append(item["result"])
            else:
                self.groups[q]["neg"].append(item["result"])

        # fallback: dataset info
        self.df = df
        self.label_map = self.build_label_map()

        self.queries = list(self.groups.keys())
        self.k_triplets = max(1, k_triplets)

    def build_label_map(self):
        label_map = {}

        for _, row in self.df.iterrows():
            label = str(row["label"])
            path = row["image_path"]

            if label not in label_map:
                label_map[label] = []

            label_map[label].append(path)

        return label_map

    def __len__(self):
        return len(self.queries) * self.k_triplets

    def __getitem__(self, idx):
        query_idx = idx // self.k_triplets
        triplet_slot = idx % self.k_triplets

        query = self.queries[query_idx]
        group = self.groups[query]

        # anchor
        anchor = self.load(query)

        # ── POSITIVE ──
        if len(group["pos"]) > 0:
            pos_path = random.choice(group["pos"])
        else:
            # fallback: same artist
            label = self.get_label(query)
            pos_path = random.choice(self.label_map[label])

        # ── NEGATIVE ──
        # When k_triplets > 1: rotate through distinct negatives from the feedback
        # group across k slots (seeded by query_idx for deterministic diversity).
        # Falls back to diverse random artists when the group has no labeled negatives.
        if len(group["neg"]) > 0:
            if self.k_triplets > 1:
                rng = random.Random(query_idx)
                candidates = list(group["neg"])
                rng.shuffle(candidates)
                neg_path = candidates[triplet_slot % len(candidates)]
            else:
                neg_path = random.choice(group["neg"])
        else:
            # fallback: different artist, diverse across k slots
            label = self.get_label(query)
            other_labels = [l for l in self.label_map.keys() if l != label]

            if self.k_triplets > 1:
                rng = random.Random(query_idx)
                rng.shuffle(other_labels)
                neg_label = other_labels[triplet_slot % len(other_labels)]
            else:
                neg_label = random.choice(other_labels)

            neg_path = random.choice(self.label_map[neg_label])

        positive = self.load(pos_path)
        negative = self.load(neg_path)

        return anchor, positive, negative

    def get_label(self, path):
        for _, row in self.df.iterrows():
            if row["image_path"] == path:
                return str(row["label"])

    def load(self, path):
        img = Image.open(os.path.join(self.root_dir, path)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img