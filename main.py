"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import torch
from architectures import SimpleCNN, DepixCNN, SimpleDeepixCNN, DeepixCNN_noskip, SimpleThickCNN, Deepixv1
import numpy as np
import matplotlib.pyplot as plt

import datasets
from utils import training_loop, check_overfitting
from data_utils import stack_with_padding

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('dataset', (4, 32), (4, 32), (4, 16))

    #model = SimpleCNN(2, 1, 14, (2, 6))
    #model = DepixCNN(2, 1, 14, (6, 2), skip_kernel=3)
    #model = SimpleDeepixCNN(2, 1, num_BasicBlocks=6, kernel_size=(6, 3))
    #model = DeepixCNN_noskip(2, 1, num_BasicBlocks=10, kernel_size=(7, 3))

    # now we start going wide:
    #model = SimpleThickCNN(2, 1, 5, (3, 6))
    model = Deepixv1(2, 1, shape=(5, 5, 6, 7), kernel_size=(3, 5))

    training_loop(model, data, 200, torch.optim.Adam, torch.nn.MSELoss(), (0.9, 0.1), 64, stack_with_padding,
                  True, True, True, 5, 'models/Deepixv1(5,5,6,7)(3,5)_scheduler.pt',
                  'losses/Deepixv1(5,5,6,7)(3,5)_scheduler.jpg', 4, True, 20)
    
    # careful to disable true_random for checking results with the following function
    check_overfitting(datasets.RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16), true_random=False),
                      "models/Deepixv1(5,5,6,7)(3,5)_scheduler.pt")
    