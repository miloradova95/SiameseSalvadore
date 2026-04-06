from torch.utils.data import DataLoader
from preprocessing.SiameseDataset import SiameseDataset

def get_dataloader(csv_path, root_dir, transform, batch_size=16, shuffle=True, mode="triplet"):
    dataset = SiameseDataset(csv_path, root_dir, transform, mode=mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )