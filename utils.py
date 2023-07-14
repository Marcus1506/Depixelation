"""
Author: Marcus Pertlwieser, 2023

Provides general utility for the project.
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pickle

from submission.submission_serialization import serialize, deserialize

def checkpoint(model: torch.nn.Module, model_path: str, device:str) -> None:
    """
    Saves a model to a given path. Brings model to CPU for saving.
    then back to device.
    """
    model.to('cpu')
    if isinstance(model_path, str):
        torch.save(model, model_path)
    model.to(device)
    return

# TODO: Add number of workers for dataloaders, add true random seed, improve early stopping by saving best model
# Consider pinning memory to CPU for dataloaders, try non_blocking=True with removing syncs in epoch loop
def training_loop(
        network: torch.nn.Module, data: torch.utils.data.Dataset, num_epochs: int,
        optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, splits: tuple[float, float],
        minibatch_size: int=16, collate_func: callable=None, show_progress: bool = False, try_cuda: bool = False,
        early_stopping: bool = True, patience: int = 3, model_path: str=None, losses_path: str=None,
        workers: int=0, pin_memory: bool=True, prefetch_factor: int=2) -> None:

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() and try_cuda else "cpu")
    print(device)
    network.to(device)

    if data.true_random:
        rng = np.random.default_rng()
        seed = rng.integers(0, 2**16 - 1, dtype=int)
    else:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Handle data
    if int(sum(splits)) != 1:
        raise ValueError("Splits must sum to 1.")
    train_data, eval_data = random_split(data, splits)

    train_dataloader = DataLoader(train_data, collate_fn=collate_func, batch_size=minibatch_size,
                                  shuffle=True, num_workers=workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    # a little less workers for eval is generally good in most cases
    eval_dataloader = DataLoader(eval_data, collate_fn=collate_func, batch_size=minibatch_size,
                                  shuffle=True, num_workers=(workers+1)//2, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

    # Hand model parameters to optimizer
    optimizer = optimizer(network.parameters())
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    training_losses = []
    eval_losses = []
    min_index = 0
    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        # set model to training mode
        network.train()
        train_minibatch_losses = []
        for train_batch, target_batch in train_dataloader:
            train_batch = train_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            # clear gradients
            network.zero_grad()

            # compute loss and propagate back
            pred = network(train_batch)
            loss = loss_function(pred, target_batch)
            loss.backward()

            # update model parameters
            optimizer.step()

            # append loss to list, detach gradients from tensor and move to cpu
            train_minibatch_losses.append(loss.detach().cpu())
        training_losses.append(torch.mean(torch.stack(train_minibatch_losses)))
        
        eval_minibatch_losses = []
        # set model to eval mode
        network.eval()
        with torch.no_grad():
            for eval_batch, target_batch in eval_dataloader:
                eval_batch = eval_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                pred = network(eval_batch)
                loss = loss_function(pred, target_batch)

                eval_minibatch_losses.append(loss.detach().cpu())
        eval_losses.append(torch.mean(torch.stack(eval_minibatch_losses)))

        if early_stopping:
            # mabye restrict the search to the last few entries
            min_index_new = eval_losses.index(min(eval_losses))
            if min_index_new != min_index: # model has improved
                checkpoint(network, model_path, device)
                # Also plot losses then!
                if isinstance(losses_path, str):
                    plot_losses(training_losses, eval_losses, losses_path)
            elif len(eval_losses) - 1 - min_index_new == patience:
                if isinstance(losses_path, str):
                    plot_losses(training_losses, eval_losses, losses_path)
                network.to('cpu')
                return
            min_index = min_index_new
        
        scheduler.step()

    checkpoint(network, model_path, device)
    if isinstance(losses_path, str):
        plot_losses(training_losses, eval_losses, losses_path)
    return

def test_loop_serialized(model_path: str, data_path: str, submission_path: str) -> None:
    """
    This function is used to test the specified model (model_path) on the provided
    pickle file, which serves as a test set. The predictions should be gathered in a
    list of 1D Numpy arrays with dtype uint8 (so rescaling necessary!). This list
    should then be serialized to file using the provided submission_serialization.py.
    """
    
    model = torch.load(model_path)
    model.eval()
    predictions = []

    with open(data_path, 'rb') as f:
        dictionary = pickle.load(f)
        with torch.no_grad():
            a=1
            for (pixelated_image, known_array) in zip(dictionary['pixelated_images'], dictionary['known_arrays']):
                input = np.concatenate((pixelated_image/255, known_array), axis=0)
                input = torch.from_numpy(input).unsqueeze(0).float()
                pred = model(input).numpy()*255
                pred = pred.astype(np.uint8)
                pred = np.squeeze(pred)
                # Prediction should only be the unknown part of the image:
                predictions.append(np.extract(known_array.flatten() == 0, pred))
    
    serialize(predictions, submission_path)

def plot_sample(data_sample: tuple[np.ndarray, np.ndarray, np.ndarray, str]) -> None:
    """
    Used for plotting samples obatained directly from the random pixelation dataset's
    __getitem__ method. The images are expected to be scaled to [0, 1].
    """
    pixelated_image, known_array, target_array, path = data_sample
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(pixelated_image[0], cmap='gray', vmin=0, vmax=1)
    axs[1].imshow(known_array[0], cmap='gray', vmin=0, vmax=1)
    axs[2].imshow(target_array[0], cmap='gray', vmin=0, vmax=1)
    plt.suptitle(path)
    plt.show()

def check_overfitting(data: torch.utils.data.Dataset, model_path: str) -> None:
    """
    Takes in dataset and model path to check if an overfitted model has
    plausible predictions on the training set.
    """
    model = torch.load(model_path)
    model.eval()
    # just to be sure:
    model.to('cpu')
    with torch.no_grad():
        for i in range(10):
            truth = data.get_image(i)
            pix = data[i][0]
            
            pred = model(torch.from_numpy(np.concatenate((data[i][0], data[i][1]), axis=0)).unsqueeze(0).float()).numpy()*255
            pred = pred.astype(np.uint8)

            fig, axs = plt.subplots(1, 3, figsize=(12, 6))
            axs[0].imshow(truth[0], cmap='gray', vmin=0, vmax=1)
            axs[1].imshow(pix[0], cmap='gray', vmin=0, vmax=1)
            visualize_flat_u8int(pred, axs[2])
            plt.show()

def plot_beatiful_samples(data: torch.utils.data.Dataset, model_path: str, indices: list[str]) -> None:
    assert len(indices) >= 2, "Please provide at least two indices!"

    model = torch.load(model_path)
    model.eval()
    # just to be sure:
    model.to('cpu')
    with torch.no_grad():
        count = 0
        fig, axs = plt.subplots(len(indices), 3, figsize=(12, 16))

        columns = ['Truth', 'Pixelated', 'Prediction']
        for ax, column in zip(axs[0], columns):
            ax.set_title(column, fontsize=14)

        for i in indices:
            truth = data.get_image(i)
            pix = data[i][0]
            
            pred = model(torch.from_numpy(np.concatenate((data[i][0], data[i][1]), axis=0)).unsqueeze(0).float()).numpy()*255
            pred = pred.astype(np.uint8)

            axs[count, 0].imshow(truth[0], cmap='gray', vmin=0, vmax=1)
            axs[count, 1].imshow(pix[0], cmap='gray', vmin=0, vmax=1)
            visualize_flat_u8int(pred, axs[count, 2])
            count += 1
    #fig.tight_layout()
    fig.suptitle('Truth/Pixelated/Prediction samples of the final model', fontsize=18)
    plt.savefig('final_model_performance/beautiful_samples.jpg')
    plt.show()
        

def plot_losses(training_losses: list[float], eval_losses: list[float], path: str) -> None:
    """
    Takes in training losses and evaluation losses and saves plots to path-directory.
    """
    plt.plot(training_losses, label='Train loss')
    plt.plot(eval_losses, label='Evaluation loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(path)

def visualize_flat_u8int(image: np.ndarray, ax) -> None:
    """
    Takes in a 1D Numpy array of type uint8 and visualizes it as a grayscale image
    on a matplotlib subplot axis.
    """
    dim = image.shape[-1]
    sqrt_dim = int(np.sqrt(dim))
    try:
        img = image.copy().reshape((int(sqrt_dim), int(sqrt_dim)))
    except:
        raise ValueError("Image must be square.")
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)

def kernel_interp(range: tuple[int, int], num:int, hidden_layers:int) -> int:
    """
    Takes in num and interpolates a fitting integer into the range based on step num.
    Always return odd, since same 
    """
    interp = int(range[0] + (range[1] - range[0]) * (float(num) / (hidden_layers - 1.)))
    if interp % 2 == 0:
        return interp - 1
    else:
        return interp

def feature_class(order: int) -> int:
    return 2**order

if __name__ == '__main__':
    # Test set prediction and serialization:
    test_loop_serialized("models_serious/Deepixv1(5,5,6,6,7,7,8)(3,5).pt", "submission/test_set.pkl", "submission/submission_5.pkl")
    pass