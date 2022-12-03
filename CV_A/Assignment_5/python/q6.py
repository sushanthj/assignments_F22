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

from torch.utils.tensorboard import SummaryWriter

def main():

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
    test_data = scipy.io.loadmat('../data/nist36_test.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    test_x, test_y = test_data['test_data'], test_data['test_labels']

    """
    previously if we had 100 examples and 4 possible classes, we got 100x4 matrix of labels
    where each row would be [0,1,0,0] meaning that the correct class label = 2 
    (as only 2nd index is equal to 1)

    Now, pytorch does not like this format. If there are 100 examples, pytorch wants a 100x1
    vector where the element in each row is the correct class directly

    To do this, we can just take the argmax of each row of our train_y, valid_y and test_y matrices

    the train_x has a shape of (10800,1024) viz 10800 images of 32x32 = 1024 size each
    the train_y has a shape of (10800, 36) viz 10800 images and 36 possible labels for each
    Note. out of the 36 possible labels for each image 
    only 1 element will be = 1, remainaing = 0
    """

    # convert the labels to a Nx1 format (see comments above)
    # train_y = np.expand_dims(np.argmax(train_y, axis=1), axis=1)
    # valid_y = np.expand_dims(np.argmax(valid_y, axis=1), axis=1)
    # test_y = np.expand_dims(np.argmax(test_y, axis=1), axis=1)

    train_y_arg = np.argmax(train_y, axis=1)
    valid_y_arg = np.argmax(valid_y, axis=1)
    test_y_arg = np.argmax(test_y, axis=1)

    # convert every image and label to a torch tensor
    train_xt = torch.from_numpy(train_x)
    train_yt = torch.from_numpy(train_y_arg)
    val_xt = torch.from_numpy(valid_x)
    val_yt = torch.from_numpy(valid_y_arg)
    test_xt = torch.from_numpy(test_x)
    test_yt = torch.from_numpy(test_y_arg)

    # check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    max_iters = 5
    # pick a batch size, learning rate
    batch_size = 100
    learning_rate = 1e-3

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
            inputs = inputs.to(torch.float32)
            # labels = labels.to(torch.float32)

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
                
                # ...log the running loss
                writer.add_scalar('training accuracy',
                                 (100 * correct // total),
                                epoch * len(train_loader) + i)
                running_loss = 0.0



    print('Finished Training')

    # save the trained model to disk
    PATH = './sush_net.pth'
    torch.save(net.state_dict(), PATH)

    
    
    # reload the network to measure test accuracy
    net = SushNet()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    """
    # USING NUMPY
    # Run Validation Accuracy pass

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs, labels = val_xt.to(device), val_yt.to(device)
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        
        # calculate outputs by running images through the network
        outputs = net(inputs)
        outputs = outputs.cpu()
        # the class with the highest energy is what we choose as prediction
        
        nump_outputs = outputs.numpy()
        print("outputs shape is", nump_outputs.shape)

        labels = labels.cpu()
        nump_labels = labels.numpy()
        print("labels size is", labels.shape)

        loss, acc = compute_loss_and_acc(valid_y, nump_outputs)
        print("Validation Loss is", loss)
        print("Validation Accuracy is", acc)
    """
    
    # Run Validation Accuracy Pass
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs, labels = val_xt.to(device), val_yt.to(device)
        inputs = inputs.to(torch.float32)
        # labels = labels.to(torch.float32)
        
        # calculate correct label predictions
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Validation accuracy is", (100 * correct // total))

    
    # Run Test Accuracy Pass
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs, labels = test_xt.to(device), test_yt.to(device)
        inputs = inputs.to(torch.float32)
        # labels = labels.to(torch.float32)
        
        # calculate outputs by running images through the network
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Test accuracy is", (100 * correct // total))


def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    # probs is the output of our softmax function
    # we'll define our loss as the -1*(unnormalized log of these probs)

    assert y.shape == probs.shape
    log_probs = y * (probs)
    loss = -(np.sum(log_probs))

    # calculate accuracy over all training examples
    true_positives = (np.where(np.argmax(y, axis=1) == np.argmax(probs, axis=1)))[0].shape[0]
    acc = true_positives / probs.shape[0]

    return loss, acc 


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

if __name__ == '__main__':
    main()