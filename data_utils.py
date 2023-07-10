"""
Author: Marcus Pertlwieser, 2023

Provides data utility for the project.
"""

import numpy as np
import random
import torch

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    """Converts a PIL image to grayscale."""

    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255
    
    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]

def prepare_image(
        image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Takes in image and returns pixelated_image, known_array and target_array based
    on specified values."""

    if image.ndim < 3 or image.shape[-3] != 1:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (..., 1, H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")
    
    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (..., slice(y, y + height), slice(x, x + width))
    
    # This returns already a copy, so we are independent of "image"
    pixelated_image = pixelate(image, x, y, width, height, size)
    
    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False
    
    return pixelated_image, known_array, image

def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    """"Pixelates are defined by parameters, by replacing parts of the image by mean values of specified size."""
    # Need a copy since we overwrite data directly
    image = image.copy()
    curr_x = x
    
    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size
    
    return image

def random_det(img, index, width_range, height_range, size_range) -> tuple[int, int, int, int, int]:
    """Generates random parameters based on specified ranges for pixelation."""
    random.seed(index)
    
    width = random.randint(*width_range)
    height = random.randint(*height_range)
    
    if width > img.shape[2]:
        width = img.shape[2]
    if height > img.shape[1]:
        height = img.shape[1]
    
    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)
    
    size = random.randint(*size_range)
    return x, y, width, height, size

def stack_with_padding(batch_as_list: list) -> tuple[torch.tensor, torch.tensor]:
    """
    Handles stacking and batching of pixelated images, known arrays and targets.
    Should stack pixelated images and known arrays into first batch and targets
    """
    try:
        stacked_pix_imgs = np.array([entry[0] for entry in batch_as_list])
        stacked_known_arrays = np.array([entry[1] for entry in batch_as_list])

        stacked_input = torch.from_numpy(np.concatenate((stacked_pix_imgs, stacked_known_arrays), axis=1))

        target_arrays = torch.from_numpy(np.array([entry[2] for entry in batch_as_list]))
        return stacked_input, target_arrays
    except ValueError:
        # TODO: Change this implementation to be more performant, use numpy arrays directly
        # Most of the data will be same dimensions anyway so it should be fine
        pix_imgs = [entry[0] for entry in batch_as_list]
        known_arrays = [entry[1] for entry in batch_as_list]
        target_arrays = [entry[2] for entry in batch_as_list]

        max_height = max(pix_imgs, key=lambda x: x.shape[-2]).shape[-2]
        max_width = max(pix_imgs, key=lambda x: x.shape[-1]).shape[-1]
        
        for i, (pix_img, known_array) in enumerate(zip(pix_imgs, known_arrays)):
            pad_h = max_height - pix_img.shape[-2]
            pad_w = max_width - pix_img.shape[-1]
            
            pix_imgs[i] = np.pad(pix_img, ((0, 0), (0, pad_h),(0, pad_w)), mode='constant', constant_values=0)
            known_arrays[i] = np.pad(known_array, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=1)
            target_arrays[i] = np.pad(target_arrays[i], ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        pix_imgs = np.array(pix_imgs)
        known_arrays = np.array(known_arrays)
        target_arrays = torch.from_numpy(np.array(target_arrays))

        stacked_input = torch.from_numpy(np.concatenate((pix_imgs, known_arrays), axis=1))
        
        return stacked_input, target_arrays
    except:
        raise ValueError("Something went wrong with stacking the batch")