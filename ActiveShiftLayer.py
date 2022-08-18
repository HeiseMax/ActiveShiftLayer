import numpy as np
import torch
from torch import nn
from torch.nn import functional
import kornia


class ASL(nn.Module):
    def __init__(self, size_in, device):
        super().__init__()

        self.size_in, self.size_out = size_in, size_in

        # init shifts
        self.initial = torch.rand(
            (self.size_in, 2), requires_grad=False) * 2 - 1
        self.shifts = nn.Parameter(self.initial.clone().to(device))

    def forward(self, x):
        # swap batch and channels
        x = torch.transpose(x.float(), 0, 1)
        shifted = torch.zeros_like(x)
        shifted = kornia.geometry.transform.translate(x, self.shifts)
        shifted = torch.transpose(shifted, 0, 1)
        return shifted


class ASL2(nn.Module):
    def __init__(self, size_in, device):
        super().__init__()

        self.size_in, self.size_out = size_in, size_in

        # init shifts
        self.initial = torch.rand(
            (self.size_in, 2), requires_grad=True) * 2 - 1
        self.shifts = nn.Parameter(self.initial.clone().to(device))

    def forward(self, x):
        # swap batch and channels
        shifted = torch.zeros_like(x)
        for i in range(self.shifts.shape[1]):
            shifted[:, i] = kornia.geometry.transform.translate(
                x[:, i], self.shifts)
        return shifted


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
