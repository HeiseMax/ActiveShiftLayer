import numpy as np
import torch
from torch import nn
from torch.nn import functional, Module, Sequential, ReLU, Conv2d, BatchNorm2d, ModuleList, Identity
import kornia


def shift_img(img, s, device):
    ''' img shape : (1,1,h,w)
        s   shape : (1,2)
    '''
    frac = (s - s.type(torch.int32)).to(device)
    frac = frac[0]
    s_neg = torch.floor((frac - torch.abs(frac))/2).to(device)
    s_neg = s_neg.type(torch.int32)

    shifts = (s + s_neg).to(device)
    frac = torch.abs(frac)
    shifts = shifts.type(torch.int32)

    new_size = torch.tensor([img.size(2) + 2, img.size(3) + 2]).to(device)
    old_size = torch.tensor([img.size(2), img.size(3)]).to(device)

    fr = torch.ones((4, 1, 1, new_size[0], new_size[1])).to(device)
    fr[0] *= frac[0]*frac[1]
    fr[1] *= frac[0]*(1-frac[1])
    fr[2] *= (1-frac[0])*frac[1]
    fr[3] *= (1-frac[0])*(1-frac[1])

    f2 = torch.zeros((4, 1, 1, new_size[0], new_size[1])).to(device)
    f2[0, :, :, 1:new_size[0]-1, 1:new_size[1]-1] = img.clone()
    f2[1, :, :, 2:new_size[0], 1:new_size[1]-1] = img.clone()
    f2[2, :, :, 1:new_size[0]-1, 2:new_size[1]] = img.clone()
    f2[3, :, :, 2:new_size[0],  2:new_size[1]] = img.clone()

    temp = f2[0]*fr[3] + f2[1]*fr[1] + f2[2]*fr[2] + f2[3]*fr[0]

    f_im = torch.zeros((1, 1, old_size[0], old_size[1])).to(device)

    bound_right = torch.minimum(
        new_size[0], new_size[0] - shifts[0, 0] - (1+s_neg[0]))
    bound_left = torch.maximum(torch.tensor(1).to(device), 1 - shifts[0, 0])
    bound_up = torch.minimum(
        new_size[1], new_size[1] - shifts[0, 1] - (1+s_neg[1]))
    bound_down = torch.maximum(torch.tensor(1).to(device), 1 - shifts[0, 1])
    bound_finalxr = torch.minimum(old_size[0]+1, old_size[0] + shifts[0, 0]+1)
    bound_finalxl = torch.maximum(torch.tensor(0).to(device), shifts[0, 0])
    bound_finalyu = torch.minimum(old_size[1]+1, old_size[1] + shifts[0, 1]+1)
    bound_finalyd = torch.maximum(torch.tensor(0).to(device), shifts[0, 1])

    f_im[:, :, bound_finalxl:bound_finalxr, bound_finalyd:bound_finalyu] = temp[:,
                                                                                :, bound_left:bound_right, bound_down:bound_up]
    return f_im


class ASL_ownimpl(Module):
    def __init__(self, size_in, device):
        super().__init__()

        batchsize = 1

        self.size_in, self.size_out = size_in, size_in
        self.device = device

        # init shifts
        self.initial = (torch.rand(
            (batchsize, 2), requires_grad=True) * 2 - 1)
        self.shifts = nn.Parameter(self.initial.clone().to(device))

    def forward(self, x):
        shifted = shift_img(x, self.shifts, self.device)
        return shifted


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

class ASL_(Module):
    def __init__(self, size_in, device):
        super().__init__()

        self.size_in, self.size_out = size_in, size_in

        # init shifts
        self.initial = (torch.randn(
            (self.size_in, 2), requires_grad=True) * 2 - 1)*.5
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

    def __init__(self, input_size, output_size, kernel_size, padding, k=1):
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
