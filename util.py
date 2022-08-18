import torch
import time


def test_loss(NN, test_dataloader, criterion, device):
    correct = 0
    total = 0
    total2 = 0
    loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
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

    return loss/total2, 100 * correct / total


def train_NN(NN, train_dataloader, test_dataloader, epochs, optimizer, criterion, scheduler, device, steps_to_test, print_test=True):
    train_losses = []
    test_losses = []
    test_accuracies = []
    ex_times = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        ex_time_start = time.process_time_ns()
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = NN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % steps_to_test == steps_to_test - 1:    # print every steps_to_test mini-batches
                test_time_start = time.process_time_ns()
                train_loss = running_loss / steps_to_test
                current_test_loss, test_accuracy = test_loss(
                    NN, test_dataloader, criterion, device)
                train_losses.append(train_loss)
                test_losses.append(current_test_loss)
                test_accuracies.append(test_accuracy)
                if print_test:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] train_loss: {train_loss:.3f}')
                    print(
                        f'test_loss: {current_test_loss:.3f}, test_accuracy: {test_accuracy}')
                running_loss = 0.0
                test_time_end = time.process_time_ns()
        scheduler.step()
        ex_time_end = time.process_time_ns()
        ex_time = (ex_time_end - ex_time_start -
                   (test_time_end - test_time_start)) * 1e-9
        ex_times.append(ex_time)

    return train_losses, test_losses, test_accuracies, ex_times
