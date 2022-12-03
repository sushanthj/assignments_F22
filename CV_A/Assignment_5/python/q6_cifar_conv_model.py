import torch
import torch.nn as nn
import torch.nn.functional as F

class SushConvNet_CIFAR(nn.Module):

    def __init__(self):
        # inherit the necessary superclass
        super().__init__()
        
        # initialize layers

        # LAYER 1
        # the train_x shape of (10800,1024) will cause 
        # the weights to have shape (1024 x hidden_layer_size) 
        # (hidden layer size is arbitrary and here = 64)

        # LAYER 2
        # here too the weights will have a size which maps the hidden layer
        # to the output layer. The weights shape will be (hidden_size, 36)
        # where 36 = number of total possible labels
        self.layer1_shape = (400, 200)
        self.layer2_shape = (200, 100)

        # Define Operations
        # conv operations

        # conv2d (in_channels, out_channels, kernel size)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # maxpool2d (kernel_size, stride), automatically zero pads
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = XW + b
        self.fc1 = nn.Linear(self.layer1_shape[0], self.layer1_shape[1])
        self.fc2 = nn.Linear(self.layer2_shape[0], self.layer2_shape[1])

    def forward(self, x):
        # reshape to N,C,H,W
        x = x.view(-1, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
    # Backward function doesn't need explicit functions, Autograd will pick up
    # required gradients for each layer on its own