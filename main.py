"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import torch
from architectures import SimpleCNN, DepixCNN, SimpleDeepixCNN, DeepixCNN_noskip
import numpy as np
import matplotlib.pyplot as plt

import datasets
from utils import training_loop, check_overfitting
from data_utils import stack_with_padding

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('dataset_serious', (4, 32), (4, 32), (4, 16))

    #model = SimpleCNN(2, 1, 14, (2, 6))
    #model = DepixCNN(2, 1, 14, (6, 2), skip_kernel=3)
    #model = SimpleDeepixCNN(2, 1, num_BasicBlocks=6, kernel_size=(6, 3))
    model = DeepixCNN_noskip(2, 1, num_BasicBlocks=10, kernel_size=(7, 3))

    training_loop(model, data, 200, torch.optim.Adam, torch.nn.MSELoss(), (0.9, 0.1), 256, stack_with_padding,
                  True, True, True, 5, 'models_serious/DeepixCNN10(7,3)noskip_noinf_bigunif.pt',
                  'losses/DeepixCNN10(7,3)noskip_noinf_bigunif.jpg', 4, True, 20)
    
    # careful to disable true_random for checking results with the following function
    check_overfitting(datasets.RandomImagePixelationDataset('dataset_serious', (4, 32), (4, 32), (4, 16), true_random=False),
                      "models_serious/DeepixCNN10(7,3)noskip_noinf_bigunif.pt")
    