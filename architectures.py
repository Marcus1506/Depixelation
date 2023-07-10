"""
Author: Marcus Pertlwieser, 2023

Model Architectures for the project.
"""

import torch

class SimpleCNN(torch.nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int, num_hidden_layers: int, kernel_size: int=3,
            use_batchnorm: bool=True, padding: int='same', stride: int=1, dilation: int=1,
            activation_function: torch.nn.Module=torch.nn.ReLU) -> None:
        super().__init__()

        # Input layer
        self.input_channels = input_channels
        self.input_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
            kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.normalization = torch.nn.BatchNorm2d(input_channels)
        self.input_activation = activation_function()

        # Hidden layers
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(
                torch.nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
            if use_batchnorm:
                hidden_layers.append(torch.nn.BatchNorm2d(input_channels))
            hidden_layers.append(activation_function())
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        
        # Output layer
        self.output_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels,kernel_size=kernel_size,
                                            padding=padding, stride=stride, dilation=dilation)
        self.output_activation = activation_function()

        self.flatten = torch.nn.Flatten(start_dim=-2)

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

if __name__ == '__main__':
    model = SimpleCNN(2, 1, 5, 3)
    
    # (batch_size, channels, height, width)
    data = torch.rand((1, 2, 32, 32))

    print(model(data).shape)