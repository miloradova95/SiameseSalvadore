import torch
import pandas as pd
import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from model.SiameseNetwork import SiameseNetwork
from preprocessing.transforms import get_eval_transforms
from backend.db.chroma_client import get_chroma_client, get_or_create_collection


# ── Settings ─────────────────────────────────────────────
MODEL_PATH = "./model/biasedModel.pth"
CSV_PATH = "./dataset/processed/splits/val.csv"
IMAGE_ROOT = "./dataset/processed/images"

CHROMA_PATH = "./data/chroma_store"
COLLECTION_NAME = "paintings"

FEEDBACK_PATH = "./model/feedback.json"

NUM_QUERIES = 50
TOP_K = 10 


#  Device 
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#  Load Model 
def load_model(device):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


#  Load + Transform 
def load_and_transform(path, transform, device):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor


#  Save Feedback 
def save_feedback_batch(entries):
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.extend(entries)

    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2)


import matplotlib.cm as cm

def overlay_heatmap(image, heatmap):
    heatmap = heatmap.squeeze().cpu().numpy()

    # normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap = np.power(heatmap, 0.5)   # <--- tweak this (0.3–0.7 works well)

    # apply colormap
    colored = cm.jet(heatmap)[:, :, :3]

    colored = Image.fromarray((colored * 255).astype("uint8"))
    colored = colored.resize(image.size)

    overlay = 0.6 * np.array(image) + 0.4 * np.array(colored)
    return overlay.astype(np.uint8)


#  Visualization 
def show_result(query_img, result_img, overlay_img, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(query_img)
    axes[0].set_title("Query")
    axes[0].axis("off")

    axes[1].imshow(result_img)
    axes[1].set_title("Result")
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


#  Main 
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--artist", type=str, default=None,
                        help="Only sample queries from this artist (e.g. frida_kahlo). "
                             "Omit to sample randomly across all artists.")
    args = parser.parse_args()

    SHOW_VIS = not args.no_vis

    device = get_device()
    model = load_model(device)
    transform = get_eval_transforms()

    df = pd.read_csv(CSV_PATH)

    if args.artist:
        with open("./dataset/processed/splits/label_map.json") as f:
            label_map = json.load(f)
        artist_key = args.artist.lower().replace(" ", "_")
        if artist_key not in label_map:
            print(f"Unknown artist '{args.artist}'. Available keys: {', '.join(sorted(label_map.keys()))}")
            return
        artist_label = label_map[artist_key]["label"]
        df = df[df["label"] == artist_label]
        if df.empty:
            print(f"No images found for artist '{args.artist}' in {CSV_PATH}")
            return
        print(f"Collecting feedback for artist: {label_map[artist_key]['metadata']['name']} "
              f"({len(df)} images available)\n")

    client = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    samples = df.sample(min(NUM_QUERIES, len(df)))

    all_feedback = []

    print(f"Generating automatic feedback...\n")

    for _, row in samples.iterrows():
        query_rel = row["image_path"]
        query_label = str(row["label"])

        query_path = os.path.join(IMAGE_ROOT, query_rel)
        query_img, query_tensor = load_and_transform(query_path, transform, device)

        query_emb = model.get_embedding(query_tensor).cpu().tolist()[0]

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=TOP_K,
            include=["metadatas", "distances"]
        )

        filtered = []
        
        # Remove self-match
        for m, d in zip(results["metadatas"][0], results["distances"][0]):
            if m["image_path"] == query_rel:
                continue  # skip identical image

            filtered.append((m, d))

        filtered = filtered[:TOP_K]

        for meta, dist in filtered:
            result_rel = meta["image_path"]
            result_label = str(meta["artist"])

            result_path = os.path.join(IMAGE_ROOT, result_rel)
            result_img, result_tensor = load_and_transform(result_path, transform, device)

            # ── SFAM ──
            with torch.no_grad():
                _, _, sfam = model.forward_with_sfam(
                    query_tensor, result_tensor,
                    output_size=(224, 224)
                )

            overlay = overlay_heatmap(result_img, sfam)

            similarity = 1 - dist

            # ── AUTO LABEL ──
            label = 1 if result_label == query_label else 0

            # ── Save entry ──
            entry = {
                "query": query_rel,
                "result": result_rel,
                "label": label,
                "similarity": float(similarity)
            }

            all_feedback.append(entry)
            
            print(
            f"[FEEDBACK] Query: {query_rel} | Result: {result_rel} "
            f"| Label: {'SIMILAR' if label == 1 else 'NOT SIMILAR'} "
            f"| sim={similarity:.3f}"
)

            if SHOW_VIS:
                show_result(
                    query_img,
                    result_img,
                    overlay,
                    f"{result_label} | sim={similarity:.2f} | label={label}"
                )

    save_feedback_batch(all_feedback)

    print(f"\nSaved {len(all_feedback)} feedback samples.")


if __name__ == "__main__":
    main()