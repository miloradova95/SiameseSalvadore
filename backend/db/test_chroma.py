from chroma_client import get_chroma_client, get_or_create_collection

# --- 1. Initialize the client and collection ---
client = get_chroma_client()
collection = get_or_create_collection(client)

print(f"Collection '{collection.name}' initialized. Documents inside: {collection.count()}")

# --- 2. Insert fake embeddings ---
collection.add(
    ids=["patch_001", "patch_002", "patch_003"],
    # ids: unique string identifiers for each entry — later this will be a filename or UUID

    embeddings=[
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.5],
        [0.9, 0.8, 0.1, 0.0],
    ],

    metadatas=[
        {"source_image": "manuscript_a.jpg", "patch_index": 0},
        {"source_image": "manuscript_a.jpg", "patch_index": 1},
        {"source_image": "manuscript_b.jpg", "patch_index": 0},
    ],
    # metadatas: arbitrary key-value info attached to each vector.
    # Later this will store things like which image it came from
)

print(f"Inserted 3 fake embeddings. Total count: {collection.count()}")

# --- 3. Query: find the 2 most similar patches to a new query vector ---
query_vector = [0.1, 0.2, 0.3, 0.45]

results = collection.query(
    query_embeddings=[query_vector],
    n_results=2,
    # n_results: how many nearest neighbors to return
)

print("\n--- Query Results (most similar first) ---")
for i in range(len(results["ids"][0])):
    patch_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    meta = results["metadatas"][0][i]
    print(f"  Rank {i+1}: {patch_id} | distance: {distance:.4f} | source: {meta['source_image']}")


# Distance here is (1 - cosine_similarity).
# Lower distance = more similar. 0.0 = identical vectors.

# --- 4. Cleanup: delete test entries so the DB stays clean ---
collection.delete(ids=["patch_001", "patch_002", "patch_003"])
print(f"\nCleanup done. Collection count after delete: {collection.count()}")
