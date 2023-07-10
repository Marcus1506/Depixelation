"""
Author: Marcus Pertlwieser, 2023

Provides dataset classes for the project.
"""

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from data_utils import to_grayscale, prepare_image, random_det, stack_with_padding
from utils import plot_sample

class RandomImagePixelationDataset(Dataset):
    """Dataset class for randomly pixelated images. Currently uses indices as random seeds.
    Some starting indices seem to be off by one, needs investigation."""

    def __init__(self, image_dir: str, width_range: tuple[int, int],
                height_range: tuple[int, int], size_range: tuple[int, int],
                dtype = None):
        
        if width_range[0] < 2:
            raise ValueError("Width min value too small!")
        if height_range[0] < 2:
            raise ValueError("Height min value too small!")
        if size_range[0] < 2:
            raise ValueError("Size min value too small!")
        
        if width_range[0] > width_range[1]:
            raise ValueError("Width min bigger than max!")
        if height_range[0] > height_range[1]:
            raise ValueError("Height min bigger than max!")
        if size_range[0] > size_range[1]:
            raise ValueError("Size min bigger than max!")
        
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
        
        self.files = sorted(Path(image_dir).absolute().rglob('*.jpg'))

        # Since the same transform method was applied we will hardcode some of it:
        self.std_transforms = transforms.Compose([
            transforms.Resize(size=64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=(64, 64))])
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        img = np.array(Image.open(self.files[index]), dtype = self.dtype)
        img = to_grayscale(img)

        x, y, width, height, size = random_det(img, index, self.width_range, self.height_range, self.size_range)
        
        pixelated_image, known_array, target_array = prepare_image(img, x, y, width, height, size)
        return pixelated_image, known_array, target_array, self.files[index]
    
    def __len__(self) -> int:
        return len(self.files)

class DepixDataset(Dataset):
    def __init__(self, dir: str, dtype=None) -> None:
        self.dir = dir

        # list of absolute paths of files
        self.files = sorted(Path(self.dir).absolute().rglob('*.jpg'))
        self.dtype = dtype
        
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> np.ndarray:
        img = np.array(Image.open(self.files[index]), dtype = self.dtype)
        img = to_grayscale(img)
        return img

if __name__ == "__main__":
    data = RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16))
    
    dataloader = DataLoader(data, batch_size=5, collate_fn=stack_with_padding, shuffle=True, num_workers=0)

    for i, (stacked_input, target_array) in enumerate(dataloader):
        print(i)
        print(stacked_input.shape)
        print(target_array.shape)
    
    plot_sample(data[0])