"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import datasets
import matplotlib.pyplot as plt
from architectures import SimpleCNN
from utils import plot_sample, training_loop

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('data_sandbox', (4, 32), (4, 32), (4, 16))

    print(len(data))

    plot_sample(data[0])

