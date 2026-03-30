"""
Query ChromaDB with a painting image and visualize the most similar results.

Usage:
    python model/search.py

Set QUERY_IMAGE in CONFIG to any image path, then run.
"""

import os
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.transforms import get_eval_transforms
from model.siamese_net import SiameseNet
from backend.db.chroma_client import get_chroma_client, get_or_create_collection

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "query_image":  "dataset/archive/processed/images/rembrandt/rembrandt_0001.jpg",  # ← change this
    "checkpoint":   "model/checkpoints/best_model.pth",
    "collection":   "paintings",
    "chroma_path":  "./data/chroma_store",
    "top_k":        6,    # number of similar images to retrieve
}
# ─────────────────────────────────────────────────────────────────────────────


def embed_image(image_path: str, model, device) -> list:
    transform = get_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.get_embedding(tensor)
    return embedding.squeeze(0).cpu().tolist()


def show_results(query_path: str, results: dict, root_dir_hint: str = "."):
    """Display query image and top-k results in a matplotlib grid."""
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    n = len(metadatas)

    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5))
    fig.suptitle("Similarity Search Results", fontsize=14, fontweight="bold")

    # Query image (left column)
    query_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Result images
    for i, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        img_path = meta["image_path"]

        # Try as-is, then relative to cwd
        if not os.path.isabs(img_path):
            img_path = os.path.join(root_dir_hint, img_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (224, 224), color=(200, 200, 200))

        similarity = 1 - dist  # cosine distance → similarity
        artist = meta.get("artist", "unknown")

        axes[i].imshow(img)
        axes[i].set_title(f"{artist}\nsim={similarity:.3f}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    embedding_dim = ckpt.get("embedding_dim", 128)
    model = SiameseNet(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Connect to ChromaDB
    client = get_chroma_client(cfg["chroma_path"])
    collection = get_or_create_collection(client, cfg["collection"])

    print(f"Collection has {collection.count()} embeddings.")
    print(f"Query: {cfg['query_image']}")

    # Embed the query image
    query_embedding = embed_image(cfg["query_image"], model, device)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=cfg["top_k"],
        include=["metadatas", "distances"],
    )

    # Print results to console
    print(f"\nTop {cfg['top_k']} results:")
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
        similarity = 1 - dist
        print(f"  {i}. {meta['artist']:<30} sim={similarity:.4f}  ({meta['image_path']})")

    # Show visual grid
    show_results(cfg["query_image"], results, root_dir_hint="dataset/archive/processed/images")


if __name__ == "__main__":
    main()
