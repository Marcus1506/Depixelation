"""
Author: Marcus Pertlwieser, 2023

Model Architectures for the project.
"""

import torch

from utils import kernel_interp, feature_class

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
    After two initial convolutional layers, the input is concatenated with the output
    and another convolutional layer is applied, reducing the channel dimension back to
    the original input_channels.

    input.shape = (batch_size, input_channels, height, width)
    output.shape = (batch_size, input_channels, height, width)
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
            self.bn2 = torch.nn.BatchNorm2d(self.input_channels)
            self.bn3 = torch.nn.BatchNorm2d(self.input_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                                     padding='same', kernel_size=self.kernel_size)
        self.conv3 = torch.nn.Conv2d(in_channels=2*self.input_channels, out_channels=self.input_channels,
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
        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.relu(x)
        return x
    
class SimpleDeepixCNN(torch.nn.Module):
    """
    This architecture aims to combine SkipBlocks and BasicBlocks.
    It concatenates num_BasicBlocks BasicBlocks and then passes the output
    to one SkipBlock, from which a copy of the original input is skipped to.
    Also, kernel interpolation is applied, meaning that kernel size changes
    according to model depth (see kernel_interp).
    """
    def __init__(
            self, input_channels: int, output_channels: int, num_BasicBlocks: int,
            kernel_size: tuple[int, int]=(3,3), use_batchnorm: bool=True) -> None:
        super().__init__()

        # Input Block
        self.input_block = BasicBlock(input_channels=input_channels, use_batchnorm=use_batchnorm,
                                      kernel_size=kernel_size[0])

        basic_blocks = []
        for _ in range(num_BasicBlocks):
            basic_blocks.append(BasicBlock(input_channels=input_channels, use_batchnorm=use_batchnorm,
                                          kernel_size=kernel_interp(kernel_size, _, num_BasicBlocks)))
        self.basic_blocks = torch.nn.Sequential(*basic_blocks)

        self.skip_block = SkipBlock(input_channels=2*input_channels, output_channels=output_channels,
                                    use_batchnorm=use_batchnorm, kernel_size=kernel_size[0])
        
        #self.skip_block2 = SkipBlock(input_channels=input_channels+output_channels, output_channels=output_channels,
        #                            use_batchnorm=use_batchnorm, kernel_size=kernel_size[1])
        
        self.flatten = torch.nn.Flatten(start_dim=-2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.basic_blocks(input)
        x = self.skip_block(input, x)
        #x = self.skip_block2(input, x)
        # input.shape = (batch_size, 2, height, width)
        x = torch.where(input[:, 1, :, :] == 0., x[:, 0, :, :], input[:, 0, :, :])
        x = torch.unsqueeze(x, dim=1)
        x = self.flatten(x)
        return x

class DeepixCNN_noskip(torch.nn.Module):
    """
    Trying BasicBlocks only.
    """
    def __init__(
            self, input_channels: int, output_channels: int, num_BasicBlocks: int,
            kernel_size: tuple[int, int]=(3,3), use_batchnorm: bool=True,
            padding: str='same') -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_BasicBlocks = num_BasicBlocks
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm
        self.padding = padding

        basic_blocks = []
        for _ in range(self.num_BasicBlocks):
            basic_blocks.append(BasicBlock(input_channels=self.input_channels, use_batchnorm=self.use_batchnorm,
                                          kernel_size=kernel_interp(self.kernel_size, _, self.num_BasicBlocks)))
        self.basic_blocks = torch.nn.Sequential(*basic_blocks)

        # unification block, decided to use big kernel
        self.unification_block = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels,
                                                 padding=self.padding, kernel_size=self.kernel_size[0])

        self.flatten = torch.nn.Flatten(start_dim=-2)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.basic_blocks(input)
        x = self.unification_block(x)
        # input.shape = (batch_size, 2, height, width)
        x = torch.where(input[:, 1, :, :] == 0., x[:, 0, :, :], input[:, 0, :, :])
        x = torch.unsqueeze(x, dim=1)
        x = self.flatten(x)
        return x

class SimpleThickCNN(torch.nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int, length: int,
            kernel_size: tuple[int, int]=(3,3), use_batchnorm: bool=True,
            padding: str='same') -> None:
        super().__init__()

        self.input_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=32,
                                           padding=padding, kernel_size=kernel_size[0])
        if use_batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        hidden_channels = []
        for _ in range(length):
            hidden_channels.append(torch.nn.Conv2d(in_channels=32, out_channels=32,
                                                   padding=padding, kernel_size=kernel_interp(kernel_size, _, length)))
            if use_batchnorm:
                hidden_channels.append(torch.nn.BatchNorm2d(32))
            hidden_channels.append(torch.nn.ReLU())
        self.hidden_channels = torch.nn.Sequential(*hidden_channels)

        self.output_layer = torch.nn.Conv2d(in_channels=32, out_channels=output_channels,
                                            padding=padding, kernel_size=kernel_size[1])
        if use_batchnorm:
            self.bn2 = torch.nn.BatchNorm2d(output_channels)
        # maybe some other activation function?
        self.relu2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(start_dim=-2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(input)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.hidden_channels(x)
        x = self.output_layer(x)
        if hasattr(self, 'bn2'):
            x = self.bn2(x)
        x = self.relu2(x)
        x = torch.where(input[:, 1, :, :] == 0., x[:, 0, :, :], input[:, 0, :, :])
        x = torch.unsqueeze(x, dim=1)
        x = self.flatten(x)
        return x

class BasicAddBlock(BasicBlock):
    def __init__(self, input_channels: int, output_channels:int, use_batchnorm: bool=True,
               kernel_size: int=3):
        super().__init__(input_channels, use_batchnorm, kernel_size)

        self.output_channels = output_channels

        # Overwrite some of the last parts:
        self.conv3 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels,
                                     padding='same', kernel_size=self.kernel_size)
        if self.use_batchnorm:
            self.bn3 = torch.nn.BatchNorm2d(self.output_channels)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = x + input
        x = self.relu(x)
        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.relu(x)
        return x

class Deepixv1(torch.nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int, shape: tuple[int, ...],
            kernel_size: tuple[int, int]=(3,3), use_batchnorm: bool=True,
            padding: str='same') -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm
        self.padding = padding

        self.input_layer = torch.nn.Conv2d(in_channels=self.input_channels, out_channels=feature_class(self.shape[0]),
                                           padding=self.padding, kernel_size=self.kernel_size[0])
        if self.use_batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(feature_class(self.shape[0]))
        self.input_activation = torch.nn.ReLU()
        
        hidden_layers = []
        for i, (_in, _out) in enumerate(zip(shape[:-1], shape[1:])):
            hidden_layers.append(BasicAddBlock(input_channels=feature_class(_in),
                                                 output_channels=feature_class(_out),
                                                 use_batchnorm=self.use_batchnorm,
                                                 kernel_size=kernel_interp(self.kernel_size, i, len(self.shape)-1)))
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        
        # The last part of the model is subject to change
        # A big kernel size here seems to yield better results
        self.output_layer = torch.nn.Conv2d(in_channels=feature_class(shape[-1]), out_channels=self.output_channels,
                                            padding=self.padding, kernel_size=self.kernel_size[1])
        if self.use_batchnorm:
            self.bn2 = torch.nn.BatchNorm2d(self.output_channels)
        # Since we work with images in the range [0, 1], we use a sigmoid activation function
        self.output_activation = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten(start_dim=-2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(input)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.input_activation(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.output_activation(x)
        x = torch.where(input[:, 1, :, :] == 0., x[:, 0, :, :], input[:, 0, :, :])
        x = torch.unsqueeze(x, dim=1)
        x = self.flatten(x)
        return x

if __name__ == '__main__':
    # Here we can look at the architectures and estimate their complexity,
    # as well as their behaviour during training.
    from torchinfo import summary

    #model = DeepixCNN_noskip(2, 1, 10, (7, 3))
    #model = SimpleThickCNN(2, 1, 5, (3, 6))
    model = Deepixv1(2, 1, shape=(5, 7, 9), kernel_size=(3, 5))

    IMG_SIZE = 64
    BATCH_SIZE = 64

    summary(model, (BATCH_SIZE, 2, IMG_SIZE, IMG_SIZE))
