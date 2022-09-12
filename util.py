import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision import transforms

from ActiveShiftLayer import CSC_block

# Util for LeNet and VGG


def test_loss(NN, test_dataloader, criterion, device):
    correct = 0
    total = 0
    total2 = 0
    loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        NN.eval()
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = NN(images)
            loss += criterion(outputs, labels).item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total2 += 1
            correct += (predicted == labels).sum().item()
        NN.train()

    return loss/total2, 100 * correct / total


def train_NN(NN, criterion, train_dataloader, test_dataloader, epochs, batches_to_test, patience, device,  print_test=True, verbose=False, p_randomTransform=0):
    batches = 0
    if len(NN.batches) > 0:
        batches = NN.batches[-1]
    NN.batches_per_epoch = len(train_dataloader)

    lr = NN.lr
    weight_decay = NN.weight_decay
    momentum = NN.momentum
    NN.p_randomTransform = p_randomTransform

    optimizer = optim.SGD(NN.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience, verbose=verbose)

    NN.train()

    randomApl = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(10, translate=(
        1/20, 1/20), scale=(0.8, 1), shear=10, interpolation=transforms.InterpolationMode.BILINEAR)]), p=p_randomTransform)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_time = 0.0

        for i, data in enumerate(train_dataloader, 0):
            batches += 1
            ex_time_start = time.process_time_ns()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # seems to apply same transformations for all images in batch
            inputs = randomApl(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = NN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute stats
            running_loss += loss.item()
            ex_time_end = time.process_time_ns()
            ex_time = (ex_time_end - ex_time_start) * 1e-9
            running_time += ex_time
            if batches % batches_to_test == 0:    # calculate every "batches_to_test" mini-batches
                NN.eval()
                train_loss = running_loss / batches_to_test

                current_test_loss, test_accuracy = test_loss(
                    NN, test_dataloader, criterion, device)

                NN.train_loss.append(train_loss)
                NN.test_loss.append(current_test_loss)
                NN.test_accuracy.append(test_accuracy)
                NN.batches.append(batches)

                # print stats
                if print_test:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] train_loss: {train_loss:.3f}')
                    print(
                        f'test_loss: {current_test_loss:.3f}, test_accuracy: {test_accuracy}')

                NN.train_time.append(running_time)
                running_time = 0.0
                running_loss = 0.0

                scheduler.step(train_loss)
                NN.train()

    NN.lr = optimizer.state_dict()["param_groups"][0]["lr"]


# Util for U-Net

def test_loss_Unet(NN, test_dataloader, criterion, device):
    correct = 0
    total = 0
    total2 = 0
    loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        NN.eval()
        i = 0
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = NN(images)
            outputs = torch.permute(outputs, (0, 2, 3, 1))
            batch_size = outputs.size()[0]
            outputs = outputs.reshape(batch_size*256*256, 3)

            labels = labels.reshape(batch_size*256*256)
            labels = labels.to(torch.long)

            loss += criterion(outputs, labels).item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total2 += 1
            correct += (predicted == labels).sum().item()
            i += 1
        NN.train()

    return loss/total2, 100 * correct / total


def train_U_NET(NN, criterion, train_dataloader, test_dataloader, epochs, batches_to_test, patience, device,  print_test=True, verbose=False, p_randomTransform=0):
    batches = 0
    if len(NN.batches) > 0:
        batches = NN.batches[-1]
    NN.batches_per_epoch = len(train_dataloader)

    lr = NN.lr
    weight_decay = NN.weight_decay
    momentum = NN.momentum
    NN.p_randomTransform = p_randomTransform

    optimizer = optim.SGD(NN.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience, verbose=verbose)

    NN.train()

    randomApl = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(10, translate=(
        1/20, 1/20), scale=(0.8, 1), shear=10, interpolation=transforms.InterpolationMode.NEAREST)]), p=p_randomTransform)
    
    running_loss = 0.0
    running_time = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_dataloader, 0):
            batches += 1
            ex_time_start = time.process_time_ns()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            transformInput = torch.cat((inputs, labels), dim=1)

            # seems to apply same transformations for all images in batch
            transformInput = randomApl(transformInput)

            inputs, labels = torch.split(
                transformInput, split_size_or_sections=[3, 1], dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = NN(inputs)
            outputs = torch.permute(outputs, (0, 2, 3, 1))
            batch_size = outputs.size()[0]
            outputs = outputs.reshape(batch_size*256*256, 3)

            labels = labels.reshape(batch_size*256*256)
            labels = labels.to(torch.long)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute stats
            running_loss += loss.item()
            ex_time_end = time.process_time_ns()
            ex_time = (ex_time_end - ex_time_start) * 1e-9
            running_time += ex_time
            if batches % batches_to_test == 0:    # calculate every "batches_to_test" mini-batches
                NN.eval()
                train_loss = running_loss / batches_to_test

                current_test_loss, test_accuracy = test_loss_Unet(
                    NN, test_dataloader, criterion, device)

                NN.train_loss.append(train_loss)
                NN.test_loss.append(current_test_loss)
                NN.test_accuracy.append(test_accuracy)
                NN.batches.append(batches)

                # print stats
                if print_test:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] train_loss: {train_loss:.3f}')
                    print(
                        f'test_loss: {current_test_loss:.3f}, test_accuracy: {test_accuracy}')

                NN.train_time.append(running_time)
                running_time = 0.0
                running_loss = 0.0

                scheduler.step(train_loss)
                NN.train()

    NN.lr = optimizer.state_dict()["param_groups"][0]["lr"]


def train_U_NET_old(NN, train_dataloader, test_dataloader, epochs, optimizer, criterion, scheduler, device, steps_to_test, print_test=True):
    for epoch in range(epochs):  # loop over the dataset multiple times
        ex_time_start = time.process_time_ns()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            NN.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = NN(inputs)
            outputs = torch.permute(outputs, (0, 2, 3, 1))
            batch_size = outputs.size()[0]
            outputs = outputs.reshape(batch_size*256*256, 3)

            labels = labels.reshape(batch_size*256*256)
            labels = labels.to(torch.long)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % steps_to_test == steps_to_test - 1:    # print every steps_to_test mini-batches
                test_time_start = time.process_time_ns()
                train_loss = running_loss / steps_to_test
                current_test_loss, test_accuracy = test_loss_Unet(
                    NN, test_dataloader, criterion, device)
                NN.train_loss.append(train_loss)
                NN.test_loss.append(current_test_loss)
                NN.test_accuracy.append(test_accuracy)
                if print_test:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] train_loss: {train_loss:.3f}')
                    print(
                        f'test_loss: {current_test_loss:.3f}, test_accuracy: {test_accuracy}')
                running_loss = 0.0
                NN.train()
                test_time_end = time.process_time_ns()
        scheduler.step()
        ex_time_end = time.process_time_ns()
        ex_time = (ex_time_end - ex_time_start -
                   (test_time_end - test_time_start)) * 1e-9
        NN.train_time.append(ex_time)

# Load Datasets


def loadMNIST(batch_size):
    # transform images into normalized tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(
        "./data/MNIST",
        download=True,
        train=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        "./data/MNIST",
        download=True,
        train=False,
        transform=transform,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    return train_dataset, train_dataloader, test_dataset, test_dataloader, classes


def loadCIFAR10(batch_size):
    # transform images into normalized tensors
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True,
                                     download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False,
                                    download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_dataset, train_dataloader, test_dataset, test_dataloader, classes

# Stats


def inference_time(NN, test_dataloader, device):
    NN.eval()
    NN.to(device)
    t_proc = np.zeros(len(test_dataloader))
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            start_time = time.process_time_ns()
            _ = NN(images)
            stop_time = time.process_time_ns()

            t_proc[i] = (stop_time - start_time)
    NN.train()
    return t_proc

# Plot


def plot_loss(NN):
    plt.plot(NN.batches, NN.train_loss, label="train_loss")
    plt.plot(NN.batches, NN.test_loss, label="test_loss")
    plt.legend()
    plt.xlabel("batches")
    plt.ylabel("cross entropy loss")
    plt.title("loss")
    plt.show()


def plot_acc(NN):
    plt.plot(NN.batches, NN.test_accuracy)
    plt.xlabel("batches")
    plt.ylabel("test accuracy")
    plt.title("test accuracy")
    plt.show()


def plot_shifts(NN):
    i = 1
    for layer in NN.NN:
        if isinstance(layer, CSC_block):
            initial_shifts = layer.NN[3].initial.detach().to("cpu").numpy()
            shifts = layer.NN[3].shifts.detach().to("cpu").numpy()
            plt.scatter(initial_shifts[:, 0], initial_shifts[:, 1])
            plt.xlabel("pixelshift x")
            plt.ylabel("pixelshift y")
            plt.title(f"initial shifts ASL-layer {i}")
            plt.show()
            plt.scatter(shifts[:, 0], shifts[:, 1])
            plt.xlabel("pixelshift x")
            plt.ylabel("pixelshift y")
            plt.title(f"final shifts ASL-layer {i}")
            plt.show()
            i += 1


def plot_GPmodel(optimizer, axis_1, axis_2, axis_3, axis_3_value, elevation=30, azim=-60):
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(
        optimizer.parameter_range[axis_1][0], optimizer.parameter_range[axis_1][1], 0.0005)
    y = np.arange(
        optimizer.parameter_range[axis_2][0], optimizer.parameter_range[axis_2][1], 0.0005)
    X, Y = np.meshgrid(x, y)

    xflat = X.flatten()
    yflat = Y.flatten()
    z = np.ones_like(xflat) * axis_3_value
    p = np.array([xflat, yflat, z])

    zs = np.array(optimizer.gpmodel.predict_noiseless(p.T)[0])
    Z = zs.reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z)

    ax.set_xlabel(axis_1)
    ax.set_ylabel(axis_2)
    ax.set_zlabel('accuracy')

    ax.set_title(f'hyperparameter GP model ({axis_3}: {axis_3_value})')

    ax.view_init(elevation, azim)

    plt.show()
