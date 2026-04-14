# SiameseSalvadore
A Basic AI-Powered Morphological Retrieval Tool for Art History. Using Siamese Networks and high-dimensional vector embeddings to identify stylistic parallels in art styles.


# Dataset Preperation and image preprocessing

1. Download the dataset from [kaggle](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time). Either the full dataset or the resized folder is enough.
2. Extract the resized folder into the dataset folder.
3. Clean the image file names by executing preprocessing/clean_filestructure.py
4. Now the previous resized folders can be deleted.
5. Create Train testing and validation split by executing preprocessing/create_splits.py
6. Run example.py to test the siamese data pipeline. This example utilizes the created helpers for image transformation and siamese image pair creation which can be used directly for training the model.
   -  Pairs are generated dynamically for training
   -  images are transformed with some randomness between epochs to improve model generalization. 

## Flow

```
Preprocess → Train → Embed → Search → Evaluate → Collect Feedback → Fine-tune → Re-evaluate
```


## Step 1 — Train

```bash
python -m model.Training
```

Trains the Siamese Network with Triplet Loss. Saves weights to `model/trainedModel.pth`.

Key options in `model/Training.py`:
- `lr` (default `1e-4`) — learning rate
- `batch_size` (default `16`)
- `epochs` (default `4`)
- `margin` in `TripletLoss` (default `0.5`) — separation margin between positive/negative pairs

---

## Step 2 — Embed

```bash
python -m model.Embedd
```

Encodes all images into 128-dim vectors and stores them in a ChromaDB vector store at `data/chroma_store/`. Re-run this whenever the model is updated.

Key options in `model/Embedd.py`:
- `BATCH_SIZE` (default `64`)
- `COLLECTION_NAME` (default `"paintings"`)

---

## Step 3 — Search

```bash
python -m model.Search
```

Queries a single image against the vector store and displays the top-K most similar paintings.

Key options in `model/Search.py`:
- `QUERY_IMAGE` — path to the image you want to search with
- `TOP_K` (default `5`) — number of results to retrieve

---

## Step 4 — Evaluate

```bash
python -m model.Evaluate
```

Evaluates retrieval quality on the test set. Reports **Precision@K** and **mAP**. Results saved to `model/eval_results/results.json`.

Key options in `model/Evaluate.py`:
- `TOP_K` (default `5`)
- `CSV_PATH` (default `test.csv`) — which split to evaluate on

- Precision@K measures how many of the top K retrieved images belong to the same artist as the query, reflecting the quality of the most relevant results.
- mAP (mean Average Precision) evaluates how well correct results are ranked overall, rewarding cases where relevant images appear earlier in the retrieval list.

---

## Step 5 — Collect Feedback

```bash
# Skip image popups (recommended for large batches)
python -m model.Collect_Feedback --no-vis

# With SFAM heatmap visualization
python -m model.Collect_Feedback
```

Samples query images, retrieves results, and automatically labels them (same artist = similar, different = not similar). Feedback is appended to `model/feedback.json`.

When visualization is enabled, each result is shown alongside a **SFAM heatmap** (Similar Feature Activation Map) — a colored overlay highlighting which spatial regions of the result image drove the similarity score. Useful for understanding what the model actually responds to.

Key options in `model/Collect_Feedback.py`:
- `NUM_QUERIES` (default `50`) — number of query images to sample
- `TOP_K` (default `6`) — results retrieved per query
- `--no-vis` — disables visualization (no window popups, much faster)

Run multiple times to accumulate feedback. At least 200 triplets are needed before fine-tuning.

---

## Step 6 — Fine-tune

```bash
python -m model.Finetune_Triplet
```

Fine-tunes the model on collected feedback triplets. Requires ≥200 triplets in `model/feedback.json`. Saves the updated model to `model/fineTunedModel.pth`.

Key options in `model/Finetune_Triplet.py`:
- `EPOCHS` (default `3`)
- `LR` (default `1e-6`) — intentionally low to avoid overwriting learned features
- `BATCH_SIZE` (default `16`)

After fine-tuning, update `MODEL_PATH` in `Embedd.py` to point to `fineTunedModel.pth`, re-run **Step 2** and **Step 4** to measure improvement.

---

# Results:

For feedback collection, we used 50 query images with TOP_K = 10, resulting in approximately 100 triplets for fine-tuning. The model was then fine-tuned for 3 epochs with a batch size of 16 and a learning rate of 1e-6.

Before fine-tuning, the model achieved a Precision@5 of 0.576 and an mAP of 0.941 on the test set. After fine-tuning, performance slightly decreased to a Precision@5 of 0.558 and an mAP of 0.915.

This indicates that, with the current amount of feedback data, fine-tuning did not improve retrieval performance and instead led to a slight degradation. A likely reason is the relatively small number of training triplets and the limited diversity of feedback, which can lead to unstable updates in metric learning settings.