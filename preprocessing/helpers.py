from torch.utils.data import DataLoader
from preprocessing.SiameseDataset import SiameseDataset

def get_dataloader(csv_path, root_dir, transform, batch_size=16, shuffle=True, mode="triplet", k_triplets=1):
    dataset = SiameseDataset(csv_path, root_dir, transform, mode=mode, k_triplets=k_triplets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )