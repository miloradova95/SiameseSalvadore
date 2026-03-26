# backend/db/chroma_client.py

import chromadb

def get_chroma_client(persist_path: str = "./data/chroma_store") -> chromadb.ClientAPI:
    """
    Creates and returns a persistent ChromaDB client.

    'Persistent' means the database is saved to disk at `persist_path`.
    Every time you restart the program, your data is still there.
    Alternative: chromadb.EphemeralClient() — in-memory only, data lost on restart.
    We use persistent because we'll be storing real embeddings we want to keep.
    """
    client = chromadb.PersistentClient(path=persist_path)
    return client


def get_or_create_collection(client: chromadb.ClientAPI, name: str = "paintings"):
    """
    Returns an existing collection by name, or creates it if it doesn't exist.

    A 'collection' in ChromaDB is like a table in a traditional database —
    it groups related vectors together. All paintings embeddings go in one collection.

    metadata={"hnsw:space": "cosine"}
        Tells ChromaDB to use cosine similarity when comparing vectors.
        Cosine similarity measures the angle between two vectors, not their length.
        This is standard for comparing neural network embeddings because
        what matters is the direction of the vector, not its magnitude.
    """
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection
