"""
Author: Marcus Pertlwieser, 2023

Provides general utility for the project.
"""

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

def training_loop(
        model: torch.nn.Module, criterion, optimizer: torch.optim.Optimizer, train_loader: DataLoader,
        val_loader: DataLoader, epochs: int=5, device: str='cpu') -> tuple(list[float], list[float]):
    pass
