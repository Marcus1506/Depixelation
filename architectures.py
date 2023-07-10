"""
Author: Marcus Pertlwieser, 2023

Model Architectures for the project.
"""

import torch

class SimpleCNN(torch.nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int, hidden_layers: int, kernel_size: int=3,
            use_batchnorm: bool=True, padding: int=0, padding_mode: str='same', stride: int=1, dilation: int=1,
            activation_function: torch.nn.Module=torch.nn.ReLU) -> None:
        super().__init__()

        # Input layer
        self.input_channels = input_channels
        self.input_layer = torch.nn.Conv2d(input_channels=input_channels, output_channels=input_channels,
            kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride, dilation=dilation)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.normalization = torch.nn.BatchNorm2d(input_channels)
        self.input_activation = activation_function()

        # Hidden layers
        hidden_layers = []
        for _ in range(hidden_layers):
            hidden_layers.append(
                torch.nn.Conv2d(input_channels=input_channels, output_channels=input_channels,
                kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride, dilation=dilation))
            if use_batchnorm:
                hidden_layers.append(torch.nn.BatchNorm2d(input_channels))
            hidden_layers.append(activation_function())
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        
        # Output layer
        self.output_layer = torch.nn.Conv2d(input_channels=input_channels, output_channels=output_channels,kernel_size=kernel_size,
                                            padding=padding, padding_mode=padding_mode, stride=stride, dilation=dilation)
        self.output_activation = activation_function()

        self.flatten = torch.nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(input)
        if self.use_batchnorm:
            x = self.normalization(x)
        x = self.input_activation(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        x = self.flatten(x)
        return x
