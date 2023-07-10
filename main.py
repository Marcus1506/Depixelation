"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import torch
from architectures import SimpleCNN

import datasets
from utils import plot_sample, training_loop
from data_utils import stack_with_padding

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16))

    model = SimpleCNN(2, 1, 5, 3)

    training_loop(model, data, 10, torch.optim.Adam, torch.nn.MSELoss(), (0.8, 0.2), 4, stack_with_padding,
                  True, True, True, 4, 42, 'models/SimpleCNN_sandbox.pt', 'losses/SimpleCNN_sandbox.jpg')
    
    


