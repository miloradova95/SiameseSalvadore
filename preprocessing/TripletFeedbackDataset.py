import os
import json
from torch.utils.data import Dataset
from PIL import Image
import random

class TripletFeedbackDataset(Dataset):
    def __init__(self, json_path, root_dir, df, transform=None):
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
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
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
        if len(group["neg"]) > 0:
            neg_path = random.choice(group["neg"])
        else:
            # fallback: random other artist
            label = self.get_label(query)

            other_labels = list(self.label_map.keys())
            other_labels.remove(label)

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