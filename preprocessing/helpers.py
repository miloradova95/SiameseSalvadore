from torch.utils.data import DataLoader

from SiameseDataset import SiameseDataset

def get_dataloader(csv_path, root_dir, transform, batch_size=32, shuffle=True):
    dataset = SiameseDataset(csv_path, root_dir, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )