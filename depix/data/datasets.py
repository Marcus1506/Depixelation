"""
Author: Marcus Pertlwieser, 2023

"""

from torch.utils.data import Dataset
from pathlib import Path

class DepixDataset(Dataset):
    def __init__(self, dir: str) -> None:
        self.dir = dir

        # list of absolute paths of files
        self.files = sorted(Path(dir).absolute().rglob('*.jpg'))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        pass
