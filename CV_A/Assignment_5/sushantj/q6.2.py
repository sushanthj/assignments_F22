import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from NN import *
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# add models folder to path
sys.path.insert(0,'/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_5/python/network_models')

from q6_flower_model import SushConvNet_Flower

def main():

    # check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    max_iters = 15
    # pick a batch size, learning rate
    batch_size = 128
    learning_rate = 1e-4

    # The output of torchvision datasets are PILImage images of range [0, 1]
    # We transform them to Tensors of normalized range [-1, 1]
    transform = torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    torchvision.transforms.Resize((256,256))
                                   ])
    
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/flower')
    data_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_5/data/oxford-flowers102/'

    trainset = torchvision.datasets.ImageFolder(
                                                root=os.path.join(data_path, 'train'),
                                                transform=transform
                                               )
    train_loader = DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
    
    valset = torchvision.datasets.ImageFolder(
                                               root=os.path.join(data_path, 'val'),
                                               transform=transform
                                              )
    val_loader = DataLoader(valset, batch_size=batch_size, \
                                shuffle=True, num_workers=4, pin_memory=True)
    
    testset = torchvision.datasets.ImageFolder(
                                               root=os.path.join(data_path, 'test'),
                                               transform=transform
                                              )
    test_loader = DataLoader(testset, batch_size=batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)

    net = SushConvNet_Flower()
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

                print(f"traning accuracy for epoch {epoch} \
                        and batch {i} is {(100 * correct // total)}")\
                
                # ...log the running accuracy
                writer.add_scalar('training accuracy',
                                 (100 * correct // total),
                                epoch * len(train_loader) + i)
                running_loss = 0.0
            
        #================================== Run Val accuracy pass ===============================#
        correct_acc = 0
        total_acc = 0
        # again no gradients needed
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predictions = torch.max(outputs, 1)
                total_acc += labels.size(0)
                correct_acc += (predictions == labels).sum().item()

        # print overall accuracy
        print(f'Accuracy of the network on val images for epoch{epoch}: {100 * correct_acc // total_acc} %')
        # ...log the running accuracy
        writer.add_scalar('validation accuracy',
                            (100 * correct_acc // total_acc),
                            epoch)
        running_loss = 0.0

    print('Finished Training')

    # save the trained model to disk
    PATH = './flower_net.pth'
    torch.save(net.state_dict(), PATH)
    
    
    # reload the network to measure test accuracy
    net = SushConvNet_Flower()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    
    #==================================  Run Test accuracy pass  ================================#
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

    # print overall accuracy
    print(f'Accuracy of the network on test images: {100 * correct_acc // total_acc} %')

    

if __name__ == '__main__':
    main()