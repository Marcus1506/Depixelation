"""
Author: Marcus Pertlwieser, 2023
Main file of project. Executing training loop and testing loop.
"""

import datasets
import matplotlib.pyplot as plt
from utils import plot_sample

if __name__ == '__main__':
    data = datasets.RandomImagePixelationDataset('data_sandbox', (10, 40), (10, 40), (2, 6))

    print(len(data))

    plot_sample(data[0])
    