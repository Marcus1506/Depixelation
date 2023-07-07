"""
Author: Marcus Pertlwieser, 2023
Main file of project. Handles training and testing of model as a script.
"""

import datasets

if __name__ == '__main__':
    data = datasets.DepixDataset('data_sandbox')

    print(len(data))
