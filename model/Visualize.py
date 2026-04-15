import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE

from model.SiameseNetwork import SiameseNetwork
from preprocessing.transforms import get_eval_transforms
from backend.db.chroma_client import get_chroma_client, get_or_create_collection

# ── Settings ─────────────────────────────────────────────
MODEL_FT_PATH      = "./model/fineTunedModel.pth"
CSV_PATH           = "./dataset/processed/splits/test.csv"
IMAGE_ROOT         = "./dataset/processed/images"
SAVE_PATH          = "./model/eval_results/tsne.png"

CHROMA_PATH        = "./data/chroma_store"
COLLECTION_BASE    = "paintings"           # already populated by Embedd.py
COLLECTION_FT      = "paintings_finetuned" # populate by running Embedd.py with fineTunedModel.pth

TOP_N_ARTISTS      = 15    # limit artists for readability
TSNE_PERPLEXITY    = 30
TSNE_ITERATIONS    = 1000


# ── Fetch embeddings from ChromaDB ────────────────────────
def fetch_embeddings_from_chroma(collection_name, image_paths):
    client = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, collection_name)

    results = collection.get(include=["embeddings", "metadatas"])

    total_in_db = len(results["metadatas"])
    print(f"  Collection '{collection_name}' has {total_in_db} entries")

    if total_in_db == 0:
        return np.array([]), []

    print(f"  Sample stored path : {results['metadatas'][0]['image_path']}")
    print(f"  Sample CSV path    : {image_paths[0]}")

    path_to_emb = {
        m["image_path"]: e
        for m, e in zip(results["metadatas"], results["embeddings"])
    }

    found   = [p for p in image_paths if p in path_to_emb]
    ordered = [path_to_emb[p] for p in found]

    missing = len(image_paths) - len(found)
    if missing > 0:
        print(f"  Warning: {missing} images not found in '{collection_name}'")

    return np.array(ordered), found


# ── Generate embeddings (for fine-tuned model) ────────────
def generate_embeddings(model, image_paths, transform, device):
    embeddings = []

    for img_path in tqdm(image_paths, desc="Embedding"):
        full_path = os.path.join(IMAGE_ROOT, img_path)
        img = Image.open(full_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.get_embedding(tensor).cpu().numpy().flatten()

        embeddings.append(emb)

    return np.array(embeddings)


# ── t-SNE ─────────────────────────────────────────────────
def run_tsne(embeddings):
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                max_iter=TSNE_ITERATIONS, random_state=42)
    return tsne.fit_transform(embeddings)


# ── Plot ──────────────────────────────────────────────────
def plot_tsne(coords_base, coords_ft, labels, save_path):
    unique_labels = sorted(set(labels))
    colors = cm.get_cmap("hsv", len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, coords, title in [
        (axes[0], coords_base, "Before Fine-tuning"),
        (axes[1], coords_ft,   "After Fine-tuning"),
    ]:
        for label in unique_labels:
            idx = np.array([i for i, l in enumerate(labels) if l == label])
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                color=color_map[label],
                label=label,
                s=40,
                alpha=0.7,
            )
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5,
               fontsize=8, markerscale=2, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("t-SNE of Painting Embeddings by Artist", fontsize=16, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()


# ── Model ─────────────────────────────────────────────────
def load_model(path, device):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = get_eval_transforms()

    # Load test set and filter to top N artists
    df = pd.read_csv(CSV_PATH)
    top_artists = df["label"].value_counts().head(TOP_N_ARTISTS).index
    df = df[df["label"].isin(top_artists)].reset_index(drop=True)
    image_paths = df["image_path"].tolist()
    # Extract artist name from path (e.g. "vincent_van_gogh/painting.jpg" → "vincent van gogh")
    df["artist"] = df["image_path"].apply(lambda p: p.split("/")[0].replace("_", " "))
    path_to_label = dict(zip(df["image_path"], df["artist"]))

    print(f"Artists: {TOP_N_ARTISTS}, Images: {len(df)}, Device: {device}")

    # Base embeddings — fetch from ChromaDB (already stored by Embedd.py)
    print(f"\n[1/4] Fetching base embeddings from ChromaDB ('{COLLECTION_BASE}')...")
    emb_base, found_paths = fetch_embeddings_from_chroma(COLLECTION_BASE, image_paths)

    if len(emb_base) == 0:
        raise RuntimeError(
            f"No embeddings found in '{COLLECTION_BASE}'. "
            "Run 'python -m model.Embedd' first."
        )

    labels = [path_to_label[p] for p in found_paths]

    # Fine-tuned embeddings — fetch from ChromaDB or generate on the fly
    print(f"\n[2/4] Checking for fine-tuned embeddings in ChromaDB ('{COLLECTION_FT}')...")
    emb_ft, _ = fetch_embeddings_from_chroma(COLLECTION_FT, found_paths)

    if len(emb_ft) == 0:
        print(f"  '{COLLECTION_FT}' is empty — generating from {MODEL_FT_PATH}...")
        model_ft = load_model(MODEL_FT_PATH, device)
        emb_ft = generate_embeddings(model_ft, found_paths, transform, device)

    # t-SNE
    print("\n[3/4] Running t-SNE on base embeddings...")
    coords_base = run_tsne(emb_base)
    print("[4/4] Running t-SNE on fine-tuned embeddings...")
    coords_ft = run_tsne(emb_ft)

    # Plot
    plot_tsne(coords_base, coords_ft, labels, SAVE_PATH)


if __name__ == "__main__":
    main()
