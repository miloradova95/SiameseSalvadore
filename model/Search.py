import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

from model.SiameseNetwork import SiameseNetwork
from preprocessing.transforms import get_eval_transforms
from backend.db.chroma_client import get_chroma_client, get_or_create_collection


#  Settings 
MODEL_PATH = "./model/biasedFinetunedModel.pth"
CHROMA_PATH = "./data/chroma_store"
COLLECTION_NAME = "paintings"

QUERY_IMAGE = "./dataset/processed/images/frida_kahlo/frida_kahlo_0113.jpg"
TOP_K = 5


#  Device 
def get_device():
    print("CUDA available:", torch.cuda.is_available())
    return "cuda" if torch.cuda.is_available() else "cpu"


#  Load Model 
def load_model(device):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded")
    return model


#  Embed Image 
def embed_image(image_path, model, device):
    transform = get_eval_transforms()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.get_embedding(tensor)

    return embedding.squeeze(0).cpu().tolist()


#  Search 
def search_similar(collection, query_embedding):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K + 1,  # Get one extra to exclude the query image itself
        include=["metadatas", "distances"]
    )
    # Exclude the first result (the query image itself)
    results["metadatas"][0] = results["metadatas"][0][1:]
    results["distances"][0] = results["distances"][0][1:]
    return results


#  Print Results 
def print_results(results):
    print(f"\nTop {TOP_K} results:\n")

    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
        similarity = 1 - dist
        print(f"{i}. {meta['artist']} | sim={similarity:.4f}")
        print(f"   {meta['image_path']}")


#  Visualize 
def show_results(query_path, results):
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    fig, axes = plt.subplots(1, len(metadatas) + 1, figsize=(15, 5))

    # Query image
    query_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("Query")
    axes[0].axis("off")

    # Results
    for i, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        img_path = meta["image_path"]

        # Fix relative path
        if not os.path.isabs(img_path):
            img_path = os.path.join("./dataset/processed/images", img_path)

        img = Image.open(img_path).convert("RGB")

        similarity = 1 - dist

        axes[i].imshow(img)
        axes[i].set_title(f"{meta['image_path']}\n{similarity:.2f}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


#  Main 
def main():
    device = get_device()

    model = load_model(device)

    client = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    print("Collection size:", collection.count())
    print("Query image:", QUERY_IMAGE)

    query_embedding = embed_image(QUERY_IMAGE, model, device)

    results = search_similar(collection, query_embedding)

    print_results(results)
    show_results(QUERY_IMAGE, results)


if __name__ == "__main__":
    main()