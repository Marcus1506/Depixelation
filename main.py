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
    data = datasets.RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16))

    
    model = SimpleCNN(2, 1, 8, 3)

    training_loop(model, data, 1000, torch.optim.Adam, torch.nn.MSELoss(), (0.8, 0.2), 2, stack_with_padding,
                  True, True, False, 10, 42, 'models/SimpleCNN_sandbox.pt', 'losses/SimpleCNN_sandbox.jpg')
    
    check_overfitting(data, "models/SimpleCNN_sandbox.pt")
    
    
