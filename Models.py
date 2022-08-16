from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Dropout2d, ReLU, Flatten, Linear, BatchNorm2d, AvgPool2d
from ActiveShiftLayer import ASL


class smallASL(Module):
    def __init__(self, input_shape, num_labels, p_drop=0.05):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(ASL(input_shape[1]),
                             Conv2d(3, 32, 1),
                             MaxPool2d(2, 2),
                             ASL(32),
                             Conv2d(32, 64, 1),
                             Dropout2d(p_drop),
                             MaxPool2d(2, 2),
                             Dropout2d(p_drop),
                             ASL(64),
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
    def __init__(self, input_size, output_size, expansion_rate, device = "cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(BatchNorm2d(input_size),
                             ReLU(),
                             Conv2d(input_size, int(
                                    output_size * expansion_rate), 1),
                             BatchNorm2d(int(output_size * expansion_rate)),
                             ReLU(),
                             ASL(int(output_size * expansion_rate), device),
                             Conv2d(int(output_size * expansion_rate),
                                    output_size, 1)
                             )

    def forward(self, x):
        return self.NN.forward(x)


class Cifar10_Net(Module):
    def __init__(self, input_shape, num_labels, expansion_rate=3, device = "cpu"):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.NN = Sequential(Conv2d(input_shape[1], 32, 7),
                             CSC_block(32, 64, expansion_rate, device),
                             CSC_block(64, 128, expansion_rate, device),
                             AvgPool2d(7),
                             Flatten(),
                             Linear(3 * 3 * 128, 64),
                             ReLU(),
                             Linear(64, num_labels)
                             )

    def forward(self, x):
        return self.NN.forward(x)
