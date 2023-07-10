"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import torch
from architectures import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt

import datasets
from utils import training_loop, check_overfitting
from data_utils import stack_with_padding

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('dataset/training', (4, 32), (4, 32), (4, 16))

    model = SimpleCNN(2, 1, 12, (2, 6))

    training_loop(model, data, 1000, torch.optim.Adam, torch.nn.MSELoss(), (0.9, 0.1), 32, stack_with_padding,
                  True, True, True, 4, 42, 'models/CNN_vkernel_steep.pt', 'losses/CNN_vkernel_steep.jpg', 6, True, 8)
    
    check_overfitting(datasets.RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16)), "models/CNN_vkernel_steep.pt")
    
    
