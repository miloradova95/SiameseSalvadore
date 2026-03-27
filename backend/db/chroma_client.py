import chromadb

#define function with default argument "./data/chroma_store" where data is stored

def get_chroma_client(persist_path: str = "./data/chroma_store") -> chromadb.ClientAPI: #type hint?

    client = chromadb.PersistentClient(path=persist_path) #local db stored in path
    return client # returns client object


def get_or_create_collection(client: chromadb.ClientAPI, name: str = "paintings"): #collection: is like a table of a db
    
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
        # metadata: hnsw: similarity metric defined (ANN (approx. nearest neighbor) and cosine similarity)
    )
    return collection # return collection object
