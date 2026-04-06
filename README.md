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
