import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Dropout2d, ReLU, Flatten, Linear, BatchNorm2d, AvgPool2d, Sigmoid, Tanh, Softmax, Dropout, ConvTranspose2d, ModuleDict
from ActiveShiftLayer import ASL, Convolution, CSC_block, Depth_wise_block


# LeNet

class LeNet(Module):

    def __init__(self, input_shape, num_labels):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.num_labels = num_labels

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        final_size = 16 * int(input_shape[2]*input_shape[3]/16)

        self.NN = Sequential(Conv2d(input_shape[1], 6, 5, padding="same"),
                             BatchNorm2d(6),
                             Tanh(),
                             AvgPool2d(2),
                             Conv2d(6, 16, 5, padding="same"),
                             BatchNorm2d(16),
                             Tanh(),
                             AvgPool2d(2),
                             Flatten(),
                             Linear(final_size, 120),
                             Tanh(),
                             Linear(120, 84),
                             Tanh(),
                             Linear(84, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class LeASLNet(Module):

    def __init__(self, input_shape, num_labels, device, expansion_rate=1):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.device = device
        self.expansion_rate = expansion_rate

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        final_size = 16 * int(input_shape[2]*input_shape[3]/16)

        self.NN = Sequential(Conv2d(input_shape[1], 6, 5, padding="same"),
                             BatchNorm2d(6),
                             Tanh(),
                             AvgPool2d(2),
                             CSC_block(6, 16, expansion_rate, device),
                             Tanh(),
                             AvgPool2d(2),
                             Flatten(),
                             Linear(final_size, 120),
                             Tanh(),
                             Linear(120, 84),
                             Tanh(),
                             Linear(84, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class LeDepthNet(Module):

    def __init__(self, input_shape, num_labels, device):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.device = device

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        final_size = 16 * int(input_shape[2]*input_shape[3]/16)

        self.NN = Sequential(Conv2d(input_shape[1], 6, 5, padding="same"),
                             BatchNorm2d(6),
                             Tanh(),
                             AvgPool2d(2),
                             Depth_wise_block(
                                 6, 16, 3, padding="same", device=device),
                             Tanh(),
                             AvgPool2d(2),
                             Flatten(),
                             Linear(final_size, 120),
                             Tanh(),
                             Linear(120, 84),
                             Tanh(),
                             Linear(84, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


# VGG (Visual Geometry Group)


class VGGNet(Module):
    def __init__(self, input_shape, num_labels, device):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.device = device

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        final_size = 128 * int(input_shape[2]*input_shape[3]/64)
        """ #channel * image_size * pool_reduction (1/4 * 1/4 *1/4) """

        self.NN = Sequential(Conv2d(input_shape[1], 32, 3, padding="same"),
                             ReLU(),
                             Conv2d(32, 32, 3, padding="same"),
                             ReLU(),
                             MaxPool2d(2),
                             Conv2d(32, 64, 3, padding="same"),
                             ReLU(),
                             Conv2d(64, 64, 3, padding="same"),
                             ReLU(),
                             MaxPool2d(2),
                             Conv2d(64, 128, 3, padding="same"),
                             ReLU(),
                             Conv2d(128, 128, 3, padding="same"),
                             ReLU(),
                             MaxPool2d(2),
                             Flatten(),
                             Linear(final_size, 128),
                             ReLU(),
                             Linear(128, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class VGGNet2(Module):
    def __init__(self, input_shape, num_labels, device):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.device = device

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        p_drop = 0.2  # param?

        final_size = 128 * int(input_shape[2]*input_shape[3]/64)
        """ #channel * image_size * pool_reduction (1/4 * 1/4 *1/4) """

        self.NN = Sequential(Conv2d(input_shape[1], 32, 3, padding="same"),
                             ReLU(),
                             Conv2d(32, 32, 3, padding="same"),
                             ReLU(),
                             MaxPool2d(2),
                             Dropout2d(p_drop),
                             Conv2d(32, 64, 3, padding="same"),
                             ReLU(),
                             Conv2d(64, 64, 3, padding="same"),
                             ReLU(),
                             MaxPool2d(2),
                             Dropout2d(p_drop),
                             Conv2d(64, 128, 3, padding="same"),
                             ReLU(),
                             Conv2d(128, 128, 3, padding="same"),
                             ReLU(),
                             MaxPool2d(2),
                             Dropout2d(p_drop),
                             Flatten(),
                             Linear(final_size, 128),
                             ReLU(),
                             Dropout(p_drop),
                             Linear(128, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class ASL_VGGNet(Module):
    def __init__(self, input_shape, num_labels, device, expansion_rate=1):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.device = device
        self.expansion_rate = expansion_rate

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        p_drop = 0.2

        final_size = 128 * int(input_shape[2]*input_shape[3]/64)
        """ #channel * image_size * pool_reduction (1/4 * 1/4 *1/4) """

        self.NN = Sequential(Conv2d(input_shape[1], 32, 3, padding="same"),
                             ReLU(),
                             CSC_block(32, 32, expansion_rate, device),
                             ReLU(),
                             MaxPool2d(2),
                             Dropout2d(p_drop),
                             CSC_block(32, 64, expansion_rate, device),
                             ReLU(),
                             CSC_block(64, 64, expansion_rate, device),
                             ReLU(),
                             MaxPool2d(2),
                             Dropout2d(p_drop),
                             CSC_block(64, 128, expansion_rate, device),
                             ReLU(),
                             CSC_block(128, 128, expansion_rate, device),
                             ReLU(),
                             MaxPool2d(2),
                             Dropout2d(p_drop),
                             Flatten(),
                             Linear(final_size, 128),
                             ReLU(),
                             Dropout(p_drop),
                             Linear(128, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


# U-Net

class U_Net(Module):
    def contracting_block(self, size_in, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out),
                           Conv2d(size_out, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def expanding_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           ConvTranspose2d(size_mid, size_out, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
                           )
        return block

    def finalizing_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_out, kernel_size=1),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def __init__(self, input_shape, size_out):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.size_out = size_out

        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        self.NN = ModuleDict({
            "contr_1": self.contracting_block(input_shape[1], 64, 3),
            "contr_2": self.contracting_block(64, 128, 3),
            "contr_3": self.contracting_block(128, 256, 3),
            "bottleneck": self.expanding_block(256, 512, 256, 3),
            "expand_1": self.expanding_block(512, 256, 128, 3),
            "expand_2": self.expanding_block(256, 128, 64, 3),
            "final": self.finalizing_block(128, 64, size_out, 3),
            "max_pool": MaxPool2d(2)
        })

    def forward(self, x):
        skip_1 = self.NN['contr_1'](x)
        contr_1 = self.NN["max_pool"](skip_1)
        skip_2 = self.NN["contr_2"](contr_1)
        contr_2 = self.NN["max_pool"](skip_2)
        skip_3 = self.NN["contr_3"](contr_2)
        contr_3 = self.NN["max_pool"](skip_3)

        bottleneck = self.NN["bottleneck"](contr_3)

        cat_1 = torch.cat((bottleneck, skip_3), 1)
        exp_1 = self.NN["expand_1"](cat_1)
        cat_2 = torch.cat((exp_1, skip_2), 1)
        exp_2 = self.NN["expand_2"](cat_2)
        cat_3 = torch.cat((exp_2, skip_1), 1)
        final = self.NN["final"](cat_3)

        return final


class Cifar10_Conv_Net(Module):
    def __init__(self, input_shape, num_labels, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        final_size = 32 * int(input_shape[2]*input_shape[3]/16)
        """ #channel * image_size * pool_reduction (1/4 * 1/4) """

        self.NN = Sequential(Conv2d(input_shape[1], 6, 5, padding="same"),
                             BatchNorm2d(6),
                             Tanh(),
                             AvgPool2d(2),
                             Conv2d(6, 16, 5, padding="same"),
                             BatchNorm2d(16),
                             Tanh(),
                             AvgPool2d(2),
                             Conv2d(16, 32, 3, padding="same"),
                             BatchNorm2d(32),
                             Tanh(),
                             Flatten(),
                             Linear(final_size, 240),
                             Tanh(),
                             Linear(240, 120),
                             Tanh(),
                             Linear(120, 84),
                             Tanh(),
                             Linear(84, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class Cifar10_ASL_Net(Module):
    def __init__(self, input_shape, num_labels, expansion_rate=3, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        final_size = 32 * int(input_shape[2]*input_shape[3]/16)
        """ #channel * image_size * pool_reduction (1/4 * 1/4) """

        self.NN = Sequential(Conv2d(input_shape[1], 6, 5, padding="same"),
                             BatchNorm2d(6),
                             Tanh(),
                             AvgPool2d(2),
                             CSC_block(6, 16, expansion_rate, device),
                             Tanh(),
                             AvgPool2d(2),
                             CSC_block(16, 32, expansion_rate, device),
                             Tanh(),
                             Flatten(),
                             Linear(final_size, 240),
                             Tanh(),
                             Linear(240, 120),
                             Tanh(),
                             Linear(120, 84),
                             Tanh(),
                             Linear(84, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class smallASL(Module):
    def __init__(self, input_shape, num_labels, p_drop=0.05, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(ASL(input_shape[1], device),
                             Conv2d(3, 32, 1),
                             MaxPool2d(2, 2),
                             ASL(32, device),
                             Conv2d(32, 64, 1),
                             Dropout2d(p_drop),
                             MaxPool2d(2, 2),
                             Dropout2d(p_drop),
                             ASL(64, device),
                             Conv2d(64, 128, 1),
                             MaxPool2d(2, 2),
                             Dropout2d(p_drop),
                             ReLU(),
                             Flatten(),
                             Linear(16*128, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class smallConvolution(Module):
    def __init__(self, input_shape, num_labels, p_drop=0.05):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, (5, 5), padding="same"),
                             MaxPool2d(2, 2),
                             Dropout2d(p_drop),
                             Conv2d(32, 64, 5, padding="same"),
                             MaxPool2d(2, 2),
                             Dropout2d(p_drop),
                             Conv2d(64, 128, 2, padding="same"),
                             MaxPool2d(2, 2),
                             Dropout2d(p_drop),
                             ReLU(),
                             Flatten(),
                             Linear(16*128, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class Cifar10_Net(Module):
    def __init__(self, input_shape, num_labels, expansion_rate=3, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, 7),
                             CSC_block(32, 32, expansion_rate, device),
                             CSC_block(32, 32, expansion_rate, device),
                             AvgPool2d(7),
                             Flatten(),
                             Linear(3 * 3 * 32, 64),
                             ReLU(),
                             Linear(64, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class MNIST_Net(Module):
    def __init__(self, input_shape, num_labels, expansion_rate=3, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, 1),
                             CSC_block(32, 32, expansion_rate, device),
                             CSC_block(32, 32, expansion_rate, device),
                             AvgPool2d(7),
                             Flatten(),
                             Linear(4 * 4 * 32, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class MNIST_Net2(Module):
    def __init__(self, input_shape, num_labels, expansion_rate=3, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, 1),
                             CSC_block(32, 32, expansion_rate, device),
                             CSC_block(32, 32, expansion_rate, device),
                             CSC_block(32, 32, expansion_rate, device),
                             AvgPool2d(7),
                             Flatten(),
                             Linear(4 * 4 * 32, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class MNIST_conv_Net(Module):

    def __init__(self, input_shape, num_labels):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, 1),
                             BatchNorm2d(32),
                             ReLU(),
                             Conv2d(32, 32, 3, padding="same"),
                             BatchNorm2d(32),
                             ReLU(),
                             Conv2d(32, 32, 5, padding="same"),
                             BatchNorm2d(32),
                             ReLU(),
                             AvgPool2d(7),
                             Flatten(),
                             Linear(4 * 4 * 32, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class MNIST_ownconv_Net(Module):

    def __init__(self, input_shape, num_labels, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Convolution(input_shape[1], 32, 1, device=device),
                             BatchNorm2d(32),
                             ReLU(),
                             Convolution(32, 32, 3, padding="same",
                                         device=device),
                             BatchNorm2d(32),
                             ReLU(),
                             Convolution(32, 32, 5, padding="same",
                                         device=device),
                             BatchNorm2d(32),
                             ReLU(),
                             AvgPool2d(7),
                             Flatten(),
                             Linear(4 * 4 * 32, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)
