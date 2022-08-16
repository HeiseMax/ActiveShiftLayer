import torch
from torch import nn
import kornia


class ASL(nn.Module):
    def __init__(self, size_in, device):
        super().__init__()

        self.size_in, self.size_out = size_in, size_in

        # init shifts
        self.initial = torch.rand(
            (self.size_in, 2), requires_grad=True) * 2 - 1
        self.shifts = nn.Parameter(self.initial.clone().to(device))

    def forward(self, x):
        # swap batch and channels
        x = torch.transpose(x.float(), 0, 1)
        shifted = torch.zeros_like(x)
        shifted = kornia.geometry.transform.translate(x, self.shifts)
        shifted = torch.transpose(shifted, 0, 1)
        return shifted
