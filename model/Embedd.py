import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import uuid
import os

from model.SiameseNetwork import SiameseNetwork
from preprocessing.transforms import get_eval_transforms
from backend.db.chroma_client import get_chroma_client, get_or_create_collection


# Paths / Settings 
MODEL_PATH = "./model/trainedModel.pth"
CSV_PATH = "./dataset/processed/splits/train.csv"
IMAGE_ROOT = "./dataset/processed/images"

CHROMA_PATH = "./data/chroma_store"
COLLECTION_NAME = "paintings"

BATCH_SIZE = 64


# Device
def get_device():
    print("CUDA available:", torch.cuda.is_available())
    return "cuda" if torch.cuda.is_available() else "cpu"


# Load Model
def load_model(device):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded")
    return model


# Load Data
def load_data():
    df = pd.read_csv(CSV_PATH)
    transform = get_eval_transforms()
    return df, transform


# Generate Embeddings
def generate_embeddings(model, df, transform, device):
    embeddings = []
    metadatas = []
    ids = []

    for i in tqdm(range(len(df)), desc="Embedding"):
        row = df.iloc[i]

        img_path = os.path.join(IMAGE_ROOT, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.get_embedding(image).cpu().numpy().flatten()

        embeddings.append(emb.tolist())
        metadatas.append({
            "image_path": row["image_path"],
            "artist": str(row["label"])
        })
        ids.append(str(uuid.uuid4()))

    return embeddings, metadatas, ids


#  Store in ChromaDB 
def store_embeddings(embeddings, metadatas, ids):
    client = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    batch_size = 1000 

    for i in range(0, len(ids), batch_size):
        print(f"Storing batch {i} to {i + batch_size}")

        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"Stored {len(ids)} embeddings in '{COLLECTION_NAME}'")


#  Main
def main():
    device = get_device()

    model = load_model(device)
    df, transform = load_data()

    embeddings, metadatas, ids = generate_embeddings(
        model, df, transform, device
    )

    store_embeddings(embeddings, metadatas, ids)


if __name__ == "__main__":
    main()