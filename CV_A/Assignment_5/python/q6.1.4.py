import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from mpl_toolkits.axes_grid1 import ImageGrid
from NN import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

# add models folder to path
sys.path.insert(0,'/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_5/python/network_models')
from q6_SUN_model import SushConvNet_SUN

def main():

    # check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    max_iters = 5
    # pick a batch size, learning rate
    batch_size = 100
    learning_rate = 1e-3

    # The output of torchvision datasets are PILImage images of range [0, 1]
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Resize((256,256))
                                   ])
    
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/sun')

    #! Change from datasets.SUN397 to ImageFolder
    trainset = torchvision.datasets.ImageFolder(
                                                root='/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_1/data/pytorch_struct/train',
                                                transform=transform
                                               )
    train_loader = DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
    
    testset = torchvision.datasets.ImageFolder(
                                               root='/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_1/data/pytorch_struct/valid',
                                               transform=transform
                                              )
    test_loader = DataLoader(testset, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
    
    
    classes = ('acquarium', 'desert', 'highway', 'kitchen', 'laundromat',
                'park', 'waterfall', 'windmill')

    net = SushConvNet_SUN()
    net.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # iterate over epochs (max_iters = epochs)
    for epoch in range(max_iters):  # loop over the dataset multiple times

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                
                # ...log the running loss
                writer.add_scalar('training loss',
                                running_loss / 20,
                                epoch * len(train_loader) + i)
                running_loss = 0.0

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                print(f"traning accuracy for epoch {epoch} and batch {i} is {(100 * correct // total)}")\
                
                # ...log the running accuracy
                writer.add_scalar('training accuracy',
                                 (100 * correct // total),
                                epoch * len(train_loader) + i)
                running_loss = 0.0

    print('Finished Training')

    # save the trained model to disk
    PATH = './sun_net.pth'
    torch.save(net.state_dict(), PATH)
    
    
    # reload the network to measure test accuracy
    net = SushConvNet_SUN()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    
    #=================================== Run Test Accuracy Pass ==============================
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    correct_acc = 0
    total_acc = 0
    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            total_acc += labels.size(0)
            correct_acc += (predictions == labels).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print overall accuracy
    print(f'Accuracy of the network on test images: {100 * correct // total} %')

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # =========================================================================================

if __name__ == '__main__':
    main()