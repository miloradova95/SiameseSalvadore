import os
import shutil
import unicodedata
import re
from collections import defaultdict

ROOT_DIR = "../dataset/archive/resized/resized"
OUTPUT_DIR = "../dataset/archive/processed/images"


# Step 1: Flatten nested folder (resized/resized)

def flatten_directory(root_dir):
    nested_dir = os.path.join(root_dir, "resized")

    if not os.path.exists(nested_dir):
        print("No nested folder found, skipping flattening.")
        return

    print("Flattening directory...")

    for filename in os.listdir(nested_dir):
        src = os.path.join(nested_dir, filename)
        dst = os.path.join(root_dir, filename)

        if os.path.isfile(src):
            shutil.move(src, dst)

    os.rmdir(nested_dir)
    print("Flattening done.\n")


# Step 2: Robust normalization

def normalize_text(text):
    # Fix broken encoding (mojibake)
    try:
        text = text.encode("latin1").decode("utf-8")
    except:
        pass

    # Normalize unicode (ü → u)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = text.lower()

    # Replace separators
    text = text.replace(" ", "_")
    text = text.replace("-", "_")

    # Remove junk characters
    text = re.sub(r"[^a-z0-9_\.]", "", text)

    return text


# Step 3: Extract artist name safely

def extract_artist_name(filename):
    name = os.path.splitext(filename)[0]

    parts = name.split("_")

    # Remove trailing number if present
    if parts[-1].isdigit():
        parts = parts[:-1]

    artist = "_".join(parts)

    return normalize_text(artist)

# Skip this variant of dürer since the dataset just has dublettes for dürer under this wrong spelling. 
def should_skip_artist(artist_name):
    return artist_name in {
        "albrecht_duoarer",
        "albrecht_duyoarer",
        "albrecth_duyoarer", 
    }

# Step 4: Process dataset

def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    counters = defaultdict(int)
    seen_artists = set()

    print("Processing images...\n")

    for filename in os.listdir(ROOT_DIR):

        if not filename.lower().endswith(".jpg"):
            continue

        old_path = os.path.join(ROOT_DIR, filename)

        # Normalize full filename
        clean_name = normalize_text(filename)

        # Extract artist
        artist = extract_artist_name(clean_name)
        if should_skip_artist(artist):
            print(f"Skipping known duplicate artist variant: {artist} ({filename})")
            continue

        if artist == "" or artist == "unknown":
            print(f"⚠️ Skipping invalid filename: {filename}")
            continue

        seen_artists.add(artist)

        artist_dir = os.path.join(OUTPUT_DIR, artist)
        os.makedirs(artist_dir, exist_ok=True)

        # New filename
        counters[artist] += 1
        new_filename = f"{artist}_{counters[artist]:04d}.jpg"

        new_path = os.path.join(artist_dir, new_filename)

        shutil.copy2(old_path, new_path)

        print(f"{filename} -> {artist}/{new_filename}")

    print("\nDone processing dataset.")
    print(f"\nArtists detected: {len(seen_artists)}")


# MAIN

if __name__ == "__main__":
    flatten_directory(ROOT_DIR)
    process_dataset()