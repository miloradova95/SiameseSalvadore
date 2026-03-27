import os
import random
import csv
import json
import pandas as pd
import unicodedata
import re

# CONFIG


DATASET_DIR = "../dataset/archive/processed/images"
CSV_PATH = "../dataset/archive/artists.csv"
OUTPUT_DIR = "../dataset/archive//processed/splits"

SPLIT_RATIOS = (0.7, 0.15, 0.15)
SEED = 67

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helpers

def normalize_artist_name(name):
    # normalize unicode (dürer → durer)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    name = name.lower()
    name = name.replace(" ", "_").replace("-", "_")

    name = re.sub(r"[^a-z0-9_]", "", name)
    
    if (name == "albrecht_durer"):
        return "albrecht_duerer"
    

    return name


def get_all_artists(dataset_dir):
    return sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])


def split_list(items, ratios):
    n = len(items)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    return (
        items[:train_end],
        items[train_end:val_end],
        items[val_end:]
    )


# Load metadata CSV

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)

    metadata = {}

    for _, row in df.iterrows():
        key = normalize_artist_name(row["name"])

        metadata[key] = {
            "name": row["name"],
            "genre": row.get("genre", ""),
            "nationality": row.get("nationality", ""),
            "years": row.get("years", ""),
            "paintings": int(row.get("paintings", 0))
        }

    return metadata


# Main split logic
def main():
    print("Loading metadata...")
    metadata = load_metadata(CSV_PATH)

    artists = get_all_artists(DATASET_DIR)

    label_map = {}
    train_rows, val_rows, test_rows = [], [], []

    print("\nSplitting dataset...\n")

    for idx, artist in enumerate(artists):
        artist_dir = os.path.join(DATASET_DIR, artist)

        images = [
            f for f in os.listdir(artist_dir)
            if f.lower().endswith((".jpg"))
        ]

        if len(images) < 5:
            print(f"Skipping {artist} (too few images: {len(images)})")
            continue

        random.shuffle(images)

        train, val, test = split_list(images, SPLIT_RATIOS)

        # label entry with metadata
        label_map[artist] = {
            "label": idx,
            "metadata": metadata.get(artist, {})
        }

        for img in train:
            train_rows.append([f"{artist}/{img}", idx])

        for img in val:
            val_rows.append([f"{artist}/{img}", idx])

        for img in test:
            test_rows.append([f"{artist}/{img}", idx])

        print(f"{artist}: {len(train)} / {len(val)} / {len(test)}")

        # Warn if metadata missing
        if artist not in metadata:
            print(f"No metadata found for: {artist}")

    # Save CSVs

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(rows)

    
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)
    
    write_csv(os.path.join(OUTPUT_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(OUTPUT_DIR, "val.csv"), val_rows)
    write_csv(os.path.join(OUTPUT_DIR, "test.csv"), test_rows)

    # Save label map with metadata 

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
        
    print("\nDone!")
    print(f"Train: {len(train_rows)}")
    print(f"Val:   {len(val_rows)}")
    print(f"Test:  {len(test_rows)}")

# Run

if __name__ == "__main__":
    main()