import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torchvision
from q6_model import SushNet
import torch.optim as optim
import torch.nn as nn

from mpl_toolkits.axes_grid1 import ImageGrid
from NN import *
from torch.utils.data import TensorDataset, DataLoader

def main():

    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
    test_data = scipy.io.loadmat('../data/nist36_test.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    test_x, test_y = test_data['test_data'], test_data['test_labels']

    # the train_x has a shape of (10800,1024) viz 10800 images of 32x32 = 1024 size each
    # the train_y has a shape of (10800, 36) viz 10800 images and 36 possible labels for each
    # Note. out of the 36 possible labels for each image 
    # only 1 element will be = 1, remainaing = 0

    # convert every image and label to a torch tensor
    train_xt = torch.from_numpy(train_x)
    train_yt = torch.from_numpy(train_y)
    val_xt = torch.from_numpy(valid_x)
    val_yt = torch.from_numpy(valid_y)
    test_xt = torch.from_numpy(test_x)
    test_yt = torch.from_numpy(test_y)

    # check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    max_iters = 5
    # pick a batch size, learning rate
    batch_size = 1080
    learning_rate = 0.1e-3

    # initialize your custom dataset to be used with the dataloader
    train_dataset = TensorDataset(train_xt, train_yt)
    val_dataset = TensorDataset(val_xt, val_yt)
    test_dataset = TensorDataset(test_xt, test_yt)

    # create dataloader objects for each of the above datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = SushNet()
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # iterate over epochs (max_iters = epochs)
    for epoch in range(max_iters):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # save the trained model to disk
    PATH = './sush_net.pth'
    torch.save(net.state_dict(), PATH)

    
    
    # reload the network to measure test accuracy
    net = SushNet()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    
    # Run Validation Accuracy pass
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy of the network on the 10000 test images: \
            {100 * correct // total} %')
    
    
    # Run Test Accuracy pass
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the network on the 10000 test images: \
            {100 * correct // total} %')

    #? PLOT CONFUSION MATRIX, TRAIN and VAL accuracy with time
    #? PRINT TEST ACCURACY


if __name__ == '__main__':
    main()