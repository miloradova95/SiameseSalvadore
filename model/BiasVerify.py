import os
import json
import argparse
from collections import Counter

import torch
import pandas as pd
from PIL import Image

from model.SiameseNetwork import SiameseNetwork
from preprocessing.transforms import get_eval_transforms
from backend.db.chroma_client import get_chroma_client, get_or_create_collection


# ── Settings ─────────────────────────────────────────────
LABEL_MAP_PATH = "./dataset/processed/splits/label_map.json"
CSV_PATH       = "./dataset/processed/splits/val.csv"
IMAGE_ROOT     = "./dataset/processed/images"
CHROMA_PATH    = "./data/chroma_store"
COLLECTION_NAME = "paintings"

DEFAULT_MODEL  = "./model/biasedFinetunedModel.pth"
# DEFAULT_MODEL  = "./model/rainedModel.pth"
DEFAULT_TOPK   = 10
NUM_QUERIES    = 5   # images of the target artist to query


# ── Helpers ───────────────────────────────────────────────
def load_label_map():
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    # forward: artist_key → label int
    key_to_label = {k: v["label"] for k, v in label_map.items()}

    # reverse: label id (as str, matching ChromaDB metadata) → display name
    id_to_name = {str(v["label"]): v["metadata"]["name"] for v in label_map.values()}

    return key_to_label, id_to_name


def resolve_artist(artist_arg, key_to_label):
    key = artist_arg.lower().replace(" ", "_")
    if key not in key_to_label:
        raise ValueError(
            f"Artist '{artist_arg}' not found in label map. "
            f"Try one of: {', '.join(sorted(key_to_label.keys()))}"
        )
    return key, key_to_label[key]


def load_model(model_path, device):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def embed_image(img_path, model, transform, device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.get_embedding(tensor)
    return emb.squeeze(0).cpu().tolist()


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verify retrieval results for a given artist")
    parser.add_argument("--artist", type=str, required=True,
                        help="Artist key to query, e.g. frida_kahlo")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model checkpoint to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                        help=f"Number of results per query (default: {DEFAULT_TOPK})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    key_to_label, id_to_name = load_label_map()
    artist_key, artist_label = resolve_artist(args.artist, key_to_label)
    artist_display = id_to_name[str(artist_label)]

    print(f"\nArtist : {artist_display} (label={artist_label})")
    print(f"Model  : {args.model}")
    print(f"Top-K  : {args.topk}\n")

    model     = load_model(args.model, device)
    transform = get_eval_transforms()

    df = pd.read_csv(CSV_PATH)
    artist_rows = df[df["label"] == artist_label]

    if artist_rows.empty:
        print(f"No images found for '{artist_display}' in val.csv")
        return

    samples = artist_rows.sample(min(NUM_QUERIES, len(artist_rows)), random_state=42)

    client     = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    all_result_names = []

    for _, row in samples.iterrows():
        query_rel  = row["image_path"]
        query_path = os.path.join(IMAGE_ROOT, query_rel)

        query_emb = embed_image(query_path, model, transform, device)

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=args.topk + 1,   # +1 to account for potential self-match
            include=["metadatas", "distances"]
        )

        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        # Remove self-match
        filtered = [
            (m, d) for m, d in zip(metadatas, distances)
            if m["image_path"] != query_rel
        ][:args.topk]

        print(f"Query: {query_rel}")
        print(f"{'Rank':<5} {'Similarity':>10}  Artist")
        print("-" * 45)

        for rank, (meta, dist) in enumerate(filtered, 1):
            result_name = id_to_name.get(str(meta["artist"]), f"label_{meta['artist']}")
            similarity  = 1 - dist
            marker = " <-- BIAS" if meta["artist"] != str(artist_label) and result_name != artist_display else ""
            print(f"  {rank:<3} {similarity:>10.4f}  {result_name}{marker}")
            all_result_names.append(result_name)

        print()

    # Summary
    print("=" * 45)
    print(f"Result distribution across {len(samples)} queries (top-{args.topk} each):")
    print("-" * 45)
    for name, count in Counter(all_result_names).most_common():
        bar = "#" * count
        tag = " (QUERY ARTIST)" if name == artist_display else ""
        print(f"  {count:>3}  {bar:<20}  {name}{tag}")
    print()


if __name__ == "__main__":
    main()
