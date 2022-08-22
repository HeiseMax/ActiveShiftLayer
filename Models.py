from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Dropout2d, ReLU, Flatten, Linear, BatchNorm2d, AvgPool2d, Sigmoid, Tanh, Softmax, Dropout
from ActiveShiftLayer import ASL, Convolution, CSC_block, Depth_wise_block


class LeNet(Module):

    def __init__(self, input_shape, num_labels):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

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

    def __init__(self, input_shape, num_labels, expansion_rate=1, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

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

    def __init__(self, input_shape, num_labels, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

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


class VGGNet(Module):
    def __init__(self, input_shape, num_labels, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

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
    def __init__(self, input_shape, num_labels, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        p_drop = 0.2

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
