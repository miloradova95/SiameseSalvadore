import torch
import pandas as pd
import json
import os
from tqdm import tqdm

from model.SiameseNetwork import SiameseNetwork
from preprocessing.transforms import get_eval_transforms
from backend.db.chroma_client import get_chroma_client, get_or_create_collection
from PIL import Image


# ── Settings ─────────────────────────────────────────────
MODEL_PATH = "./model/trainedModel.pth"
CSV_PATH = "./dataset/processed/splits/test.csv"
IMAGE_ROOT = "./dataset/processed/images"

CHROMA_PATH = "./data/chroma_store"
COLLECTION_NAME = "paintings"

TOP_K = 5
SAVE_PATH = "./model/eval_results/results.json"


#  Device 
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#  Load Model 
def load_model(device):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


#  Embed Image 
def embed_image(image_path, model, transform, device):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.get_embedding(tensor)

    return emb.squeeze(0).cpu().tolist()


#  Metrics 
def precision_at_k(results, true_label, k):
    true_label = str(true_label)

    correct = 0
    for i in range(k):
        if str(results[i]["artist"]) == true_label:
            correct += 1

    return correct / k


def average_precision(results, true_label, k):
    true_label = str(true_label)

    correct = 0
    total = 0
    ap = 0.0

    for i in range(k):
        total += 1

        if str(results[i]["artist"]) == true_label:
            correct += 1
            ap += correct / total

    if correct == 0:
        return 0.0

    return ap / correct


#  Evaluation 
def evaluate():
    device = get_device()
    model = load_model(device)
    transform = get_eval_transforms()

    df = pd.read_csv(CSV_PATH)

    client = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    precisions = []
    aps = []

    print("Evaluating...")

    for i in tqdm(range(len(df))):
        
        row = df.iloc[i]

        img_path = os.path.join(IMAGE_ROOT, row["image_path"])
        true_label = row["label"]
        
        # embed query
        query_emb = embed_image(img_path, model, transform, device)

        # search
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=TOP_K,
            include=["metadatas"]
        )

        metadatas = results["metadatas"][0]
        
        # compute metrics
        p = precision_at_k(metadatas, true_label, TOP_K)
        ap = average_precision(metadatas, true_label, TOP_K)

        precisions.append(p)
        aps.append(ap)

    # aggregate
    mean_precision = sum(precisions) / len(precisions)
    mean_ap = sum(aps) / len(aps)

    return {
        "precision_at_k": mean_precision,
        "mAP": mean_ap,
        "top_k": TOP_K,
        "num_samples": len(df)
    }


#  Save Results 
def save_results(results):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {SAVE_PATH}")


#  Main 
def main():
    results = evaluate()

    print("\n=== Evaluation Results ===")
    print(f"Samples: {results['num_samples']}")
    print(f"Precision@{results['top_k']}: {results['precision_at_k']:.4f}")
    print(f"mAP: {results['mAP']:.4f}")

    save_results(results)


if __name__ == "__main__":
    main()