import torch
import time


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


def train_NN(NN, train_dataloader, test_dataloader, epochs, optimizer, criterion, scheduler, device, steps_to_test, print_test=True):
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
            if i == 200:
                break
        NN.train()

    return loss/total2, 100 * correct / total


def train_U_NET(NN, train_dataloader, test_dataloader, epochs, optimizer, criterion, scheduler, device, steps_to_test, print_test=True):
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
