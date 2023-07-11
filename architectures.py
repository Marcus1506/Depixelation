"""
Author: Marcus Pertlwieser, 2023

Model Architectures for the project.
"""

import torch
from utils import kernel_interp

class SimpleCNN(torch.nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int, num_hidden_layers: int, kernel_size: tuple[int, int]=(3,3),
            use_batchnorm: bool=True, padding: int='same', stride: int=1, dilation: int=1,
            activation_function: torch.nn.Module=torch.nn.ReLU) -> None:
        super().__init__()

        # Input layer
        self.input_channels = input_channels
        self.input_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
            kernel_size=kernel_size[0], padding=padding, stride=stride, dilation=dilation)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.input_normalization = torch.nn.BatchNorm2d(input_channels)
        self.input_activation = activation_function()

        # Hidden layers
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(
                torch.nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                kernel_size=kernel_interp(kernel_size, _, num_hidden_layers),
                padding=padding, stride=stride, dilation=dilation))
            if use_batchnorm:
                hidden_layers.append(torch.nn.BatchNorm2d(input_channels))
            hidden_layers.append(activation_function())
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        
        # Output layer
        self.output_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size[1],
                                            padding=padding, stride=stride, dilation=dilation)
        if self.use_batchnorm:
            self.output_normalization = torch.nn.BatchNorm2d(output_channels)
        self.output_activation = activation_function()

        self.flatten = torch.nn.Flatten(start_dim=-2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(input)
        if self.use_batchnorm:
            x = self.input_normalization(x)
        x = self.input_activation(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        if self.use_batchnorm:
            x = self.output_normalization(x)
        x = self.output_activation(x)
        x = self.flatten(x)
        return x

class DepixCNN(SimpleCNN):
    def __init__(self, input_channels: int, output_channels: int, num_hidden_layers: int, kernel_size: tuple[int, int]=(3,3),
            use_batchnorm: bool=True, padding: int='same', stride: int=1, dilation: int=1,
            activation_function: torch.nn.Module=torch.nn.ReLU, skip_kernel: int=5) -> None:
        super().__init__(input_channels, output_channels, num_hidden_layers, kernel_size,
                         use_batchnorm, padding, stride, dilation, activation_function)
        
        self.skip_connection = torch.nn.Conv2d(in_channels=input_channels+output_channels, out_channels=output_channels,
                                               kernel_size=skip_kernel, padding=padding, stride=stride, dilation=dilation)
        if self.use_batchnorm:
            self.skip_normalization = torch.nn.BatchNorm2d(output_channels)
        self.skip_activation = activation_function()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(input)
        if self.use_batchnorm:
            x = self.input_normalization(x)
        x = self.input_activation(x)

        x = self.hidden_layers(x)

        x = self.output_layer(x)
        if self.use_batchnorm:
            x = self.output_normalization(x)
        x = self.output_activation(x)

        skip = torch.cat((x, input), dim=-3)
        x = self.skip_connection(skip)
        if self.use_batchnorm:
            x = self.skip_normalization(x)
        x = self.skip_activation(x)

        x = self.flatten(x)
        return x

class SkipBlock(torch.nn.Module):
    """
    Makes up a basic block which takes in an earlier input and the output of the
    current forward pass and concatenates them before passing them through an additional
    2-layered convolutional layer.
    """
    def __init__(self, input_channels: int, output_channels: int, use_batchnorm: bool=True,
                 kernel_size: int=3):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm

        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                                               padding='same', kernel_size=self.kernel_size)
        if self.use_batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(self.input_channels)
            self.bn2 = torch.nn.BatchNorm2d(self.output_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels,
                                     padding='same', kernel_size=self.kernel_size)

    def forward(self, input: torch.Tensor, forward_output: torch.Tensor) -> torch.Tensor:
        x = torch.cat((input, forward_output), dim=-3)
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.relu(x)
        return x

class BasicBlock(torch.nn.Module):
    """
    This building block is even more similiar to the original ResNet BasicBlocks, but
    instead of adding the image values, here they are concatenated. Adding earlier values
    as an additional channel seems to make more sense for this use case and its use
    is motivated entirely on heuristics.
    """
    def __init__(self, input_channels: int, use_batchnorm: bool=True,
                 kernel_size: int=3):
        super().__init__()

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm

        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                                     padding='same', kernel_size=self.kernel_size)
        if self.use_batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(self.input_channels)
            self.bn2 = torch.nn.BatchNorm2d(self.output_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                                     padding='same', kernel_size=self.kernel_size)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = torch.cat((input, x), dim=-3)
        x = self.relu(x)
        return x
    
class DeepixCNN(torch.nn.Module):
    """
    This architecture aims to combine SkipBlocks and BasicBlocks.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, output: torch.Tensor) -> torch.Tensor:
        pass

if __name__ == '__main__':
    model = SimpleCNN(2, 1, 5, 3)
    
    # (batch_size, input_channels, height, width)
    data = torch.rand((1, 2, 32, 32))

    # (batch_size, output_channels, height*width)
    print(model(data).shape)