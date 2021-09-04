from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    """smiles dataset."""

    def __init__(self, train_batches):
        """ """

        self.train_batches = train_batches

    def __len__(self):
        return len(self.train_batches)

    def __getitem__(self, idx):
        sample = self.train_batches[idx]
        return sample