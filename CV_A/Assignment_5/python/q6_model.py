import torch
import torch.nn as nn
import torch.nn.functional as F

class SushNet(nn.Module):

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
        self.layer1_shape = (1024, 64)
        self.layer2_shape = (64, 36)

        # Define Operations
        # an affine operation: y = XW + b
        self.fc1 = nn.Linear(self.layer1_shape[0], self.layer1_shape[1])
        self.fc2 = nn.Linear(self.layer2_shape[0], self.layer2_shape[1])

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
    # Backward function doesn't need explicit functions, Autograd will pick up
    # required gradients for each layer on its own