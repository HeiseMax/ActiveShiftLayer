from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Dropout2d, ReLU, Flatten, Linear, BatchNorm2d, AvgPool2d
from ActiveShiftLayer import ASL, Convolution


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


class CSC_block(Module):
    '''Convolution-Shift-Convolution'''

    def __init__(self, input_size, expansion_rate, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        output_size = input_size
        expanded_size = int(input_size * expansion_rate)

        self.NN = Sequential(BatchNorm2d(input_size),
                             ReLU(),
                             Conv2d(input_size, expanded_size, 1),
                             BatchNorm2d(expanded_size),
                             ReLU(),
                             ASL(expanded_size, device),
                             Conv2d(expanded_size, output_size, 1)
                             )

    def forward(self, x):
        post_block = self.NN.forward(x)
        return post_block + x


class Cifar10_Net(Module):
    def __init__(self, input_shape, num_labels, expansion_rate=3, device="cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, 7),
                             CSC_block(32, expansion_rate, device),
                             CSC_block(32, expansion_rate, device),
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
                             CSC_block(32, expansion_rate, device),
                             CSC_block(32, expansion_rate, device),
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
                             CSC_block(32, expansion_rate, device),
                             CSC_block(32, expansion_rate, device),
                             CSC_block(32, expansion_rate, device),
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
