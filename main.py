"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import torch
from architectures import SimpleCNN, DepixCNN, SimpleDeepixCNN
import numpy as np
import matplotlib.pyplot as plt

import datasets
from utils import training_loop, check_overfitting
from data_utils import stack_with_padding

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('dataset/training', (4, 32), (4, 32), (4, 16))

    #model = SimpleCNN(2, 1, 14, (2, 6))
    #model = DepixCNN(2, 1, 14, (6, 2), skip_kernel=3)
    model = SimpleDeepixCNN(2, 1, num_BasicBlocks=6, kernel_size=(6, 3))

    training_loop(model, data, 800, torch.optim.Adam, torch.nn.MSELoss(), (0.9, 0.1), 64, stack_with_padding,
                  True, True, True, 30, 'models/SimpleDeepixCNN8(6,3)_noinf.pt',
                  'losses/SimpleDeepixCNN8(6,3)_noinf.jpg', 4, True, 20)
    
    # careful to disable true_random for checking results with the following function
    check_overfitting(datasets.RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16), true_random=False),
                      "models/SimpleDeepixCNN8(6,3)_noinf.pt")
    