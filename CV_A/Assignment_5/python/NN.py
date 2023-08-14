import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    """
    Initialize weights according to xavier initialization

    Args:
        in_size (_type_): 
        out_size (_type_): _description_
        params (_type_): _description_
        name (str, optional): _description_. Defaults to ''.
    """
    b = np.zeros(out_size)
    W = np.random.randn(in_size, out_size) # since we're doing Wx multiplication
    # we need to ensure the number of columns = number of inputs (i.e. size of x)
    # however, they will be XW instead of Wx so here the order is switched
    W = W * np.sqrt(6/(in_size+out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# res is sigmoid activation function
def sigmoid(x):
    """
    Sigmoid Activation Function

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = (1/ (1 + np.exp(-(x))))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = X @ W + b

    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row

# each row of x is one example and num_columns is the number of classes
# therefore we need to normalize each row (to zero mean) and then 
# run softmax on each row
def softmax(x):
    res = None

    # shift the data in x
    shifted_x = x - np.expand_dims(np.max(x,axis=1),1)

    # create a column vector of the sum of each row
    exp_sum = np.expand_dims(np.sum(np.exp(shifted_x),axis=1), axis=1)

    # use this sum of each row to normalize the class prediction scores
    res = (1/exp_sum) * np.exp(shifted_x)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    # probs is the output of our softmax function
    # we'll define our loss as the -1*(unnormalized log of these probs)

    assert y.shape == probs.shape
    log_probs = y * np.log(probs)
    loss = -(np.sum(log_probs))

    # calculate accuracy over all training examples
    true_positives = (np.where(np.argmax(y, axis=1) == np.argmax(probs, axis=1)))[0].shape[0]
    acc = true_positives / probs.shape[0]

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    activation_derivative = activation_deriv(post_act) * delta
    
    # we do W*x + b => activation_deriv
    # Therefore, the sequence of gates are:
    # multiplication gate -> addition gate -> sigmoid_gate (or softmax)
    # the addition gate flows equal gradient to grad_b and mutliplication gate 
    grad_b = np.sum(activation_derivative, axis=0)/activation_derivative.shape[0]

    grad_W = X.T @ activation_derivative
    grad_X = activation_derivative @ W.T

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]

# remember: x = (examples, dimensions)
#           y = (examples, classes)
# therefore, we should split data along the rows

def get_random_batches(x,y,batch_size):
    batches = []
    print("x and y shape is", x.shape, y.shape)
    
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(x.shape[0])
    shuffled_x, shuffled_y =  x[p,:], y[p,:]

    num_batches = (x.shape[0]/batch_size)
    x_batches = np.split(shuffled_x, num_batches)
    y_batches = np.split(shuffled_y, num_batches)

    for xb, yb in zip(x_batches, y_batches):
        batches.append((xb,yb))

    return batches
