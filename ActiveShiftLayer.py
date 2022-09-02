import numpy as np
import torch
from torch import nn
from torch.nn import functional, Module, Sequential, ReLU, Conv2d, BatchNorm2d, ModuleList, Identity
import kornia


class ASL(Module):
    def __init__(self, size_in, device):
        super().__init__()

        self.size_in, self.size_out = size_in, size_in

        # init shifts
        self.initial = (torch.rand(
            (self.size_in, 2), requires_grad=True) * 2 - 1)
        self.shifts = nn.Parameter(self.initial.clone().to(device))

    def forward(self, x):
        # swap batch and channels
        x = torch.transpose(x.float(), 0, 1)
        shifted = torch.zeros_like(x)
        shifted = kornia.geometry.transform.translate(x, self.shifts)
        shifted = torch.transpose(shifted, 0, 1)
        return shifted


class CSC_block(Module):
    '''Convolution-Shift-Convolution'''

    def __init__(self, input_size, output_size, expansion_rate, device):
        '''input_size: input channel number'''
        super().__init__()

        expanded_size = int(input_size * expansion_rate)

        # rename to block
        self.NN = Sequential(
            Conv2d(input_size, expanded_size, 1),
            BatchNorm2d(expanded_size),
            ReLU(),
            ASL(expanded_size, device),
            Conv2d(expanded_size, input_size, 1)
        )

        if not (input_size == output_size):
            self.residual_conv = Conv2d(input_size, output_size, 1)
        else:
            self.residual_conv = Identity()

    def forward(self, x):
        residual = x
        x = self.NN.forward(x)
        x = x + residual
        x = self.residual_conv(x)
        return x


class CSC_block_res2(Module):
    '''Convolution-Shift-Convolution'''

    def __init__(self, input_size, output_size, expansion_rate, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        expanded_size = int(input_size * expansion_rate)

        # rename to block
        self.NN = Sequential(
            Conv2d(input_size, expanded_size, 1),
            BatchNorm2d(expanded_size),
            ReLU(),
            ASL(expanded_size, device),
            Conv2d(expanded_size, output_size, 1)
        )

        if not (output_size == 0):
            self.residual_conv = Conv2d(
                input_size + output_size, output_size, 1)
        else:
            self.residual_conv = Identity()

    def forward(self, x):
        residual = x
        x = self.NN.forward(x)
        x = torch.cat((x, residual), 1)
        x = self.residual_conv(x)
        return x


class CSC_block_res3(Module):
    '''Convolution-Shift-Convolution'''

    def __init__(self, input_size, output_size, expansion_rate, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        expanded_size = int(input_size * expansion_rate)

        # rename to block
        self.NN = Sequential(
            Conv2d(input_size, expanded_size, 1),
            BatchNorm2d(expanded_size),
            ReLU(),
            ASL(expanded_size, device)
        )

        self.residual_conv = Conv2d(expanded_size + input_size, output_size, 1)

    def forward(self, x):
        residual = x
        x = self.NN.forward(x)
        x = torch.cat((x, residual), 1)
        x = self.residual_conv(x)
        return x


class Depth_wise_block(Module):
    '''Depthwise-Convolution'''

    def __init__(self, input_size, output_size, kernel_size, padding, k=1, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        expanded_size = int(input_size * k)

        # rename to block
        self.NN = Sequential(
            Conv2d(input_size, expanded_size, 1),
            BatchNorm2d(expanded_size),
            ReLU(),
            Conv2d(expanded_size, expanded_size, kernel_size,
                   padding=padding, groups=expanded_size),
            Conv2d(expanded_size, output_size, 1)
        )

    def forward(self, x):
        return self.NN.forward(x)


class Convolution(nn.Module):
    def __init__(self, size_in, size_out, kernel_size, stride=1, padding=0, device="cpu"):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out
        self.stride = stride
        self.padding = padding

        # init weights
        k = 1/(size_in*kernel_size**2)
        sqrt_k = np.sqrt(k)
        self.initial = (torch.rand((size_out, size_in, kernel_size,
                        kernel_size), requires_grad=True) * 2 - 1) * sqrt_k
        self.weights = nn.Parameter(self.initial.clone().to(device))

        # init bias
        self.initial_bias = (torch.rand(
            (size_out), requires_grad=True) * 2 - 1) * sqrt_k
        self.bias = nn.Parameter(self.initial_bias.clone().to(device))

    def forward(self, x):
        return functional.conv2d(x, self.weights, bias=self.bias, stride=self.stride, padding=self.padding)
