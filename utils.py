"""
Author: Marcus Pertlwieser, 2023

Provides general utility for the project.
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

# TODO: Add minibacht size, hand over optimizer and loss instance directly
def training_loop(
        network: torch.nn.Module, train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset, num_epochs: int,
        show_progress: bool = False, try_cuda: bool = False,
        early_stopping: bool = True, patience: int = 3) -> tuple[list, list]:

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() and try_cuda else "cpu")
    network.to(device)

    # Maybe set seed here and use shuffling if needed.
    # Also memory could be pinned on CPU
    # to make training less prone to performance problems coming
    # from potential disk I/O operations.
    train_dataloader = torch.utils.data.DataLoader(train_data)
    eval_dataloader = torch.utils.data.DataLoader(eval_data)

    # hand the optimizer model parameters
    optimizer = torch.optim.Adam(network.parameters())
    # instantiate loss
    loss_funtion = torch.nn.MSELoss()

    training_losses = []
    eval_losses = []
    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        # set model to training mode
        network.train()
        train_minibatch_losses = []
        for train_batch, target_batch in train_dataloader:
            train_batch = train_batch.to(device)
            target_batch = target_batch.to(device)

            # clear gradients
            network.zero_grad()

            # compute loss and propagate back
            pred = network(train_batch)
            loss = loss_funtion(torch.squeeze(pred, dim=0), target_batch)
            loss.backward()

            # update model parameters
            optimizer.step()

            # append loss to list, detach gradients from tensor and move to cpu
            train_minibatch_losses.append(loss.detach().cpu())
        training_losses.append(torch.mean(torch.stack(train_minibatch_losses)))
        
        eval_minibatch_losses = []
        # set model to eval mode
        network.eval()
        for eval_batch, target_batch in eval_dataloader:
            eval_batch = eval_batch.to(device)
            target_batch = target_batch.to(device)

            pred = network(eval_batch)
            loss = loss_funtion(torch.squeeze(pred, dim=0), target_batch)

            eval_minibatch_losses.append(loss.detach().cpu())
        eval_losses.append(torch.mean(torch.stack(eval_minibatch_losses)))

        if early_stopping:
            # mabye restrict the search to the last few entries
            min_index = eval_losses.index(min(eval_losses))
            if len(eval_losses) - 1 - min_index == patience:
                # for completeness sake maybe also send network back to cpu here 
                return training_losses, eval_losses

    # change device back to cpu
    network.to('cpu')

    return training_losses, eval_losses