import numpy as np
# you should write your functions in nn.py
from NN import *
from util import *

def main():
    # fake data
    # feel free to plot it in 2D
    # what do you think these 4 classes are?
    g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
    g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
    g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
    g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
    x = np.vstack([g0,g1,g2,g3])
    # we will do XW + B
    # that implies that the data is N x D

    # create labels
    y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
    # turn to one_hot
    y = np.zeros((y_idx.shape[0],y_idx.max()+1))
    y[np.arange(y_idx.shape[0]),y_idx] = 1

    # parameters in a dictionary
    params = {}

    print("STARTING X shape is", x.shape)
    print("STARTING Y shape is", y.shape)

    # Q 2.1
    # initialize a layer
    initialize_weights(2,25,params,'layer1')
    initialize_weights(25,4,params,'output')
    assert(params['Wlayer1'].shape == (2,25))
    assert(params['blayer1'].shape == (25,))

    # expect 0, [0.05 to 0.12]
    print("{}, {:.2f}".format(params['blayer1'].mean(),params['Wlayer1'].std()**2))
    print("{}, {:.2f}".format(params['boutput'].mean(),params['Woutput'].std()**2))


    # Q 2.2.1
    # implement sigmoid
    test = sigmoid(np.array([-1000,1000]))
    print('should be zero and one\t',test.min(),test.max())

    # Forward function does the following in order:
    # 1. Finds the value of XW + b (idk why it's XW, this was TA's choice)
    # 
    h1 = forward(x,params,'layer1', sigmoid)
    print("forward shape is",h1.shape)

    # Q 2.2.2
    # implement softmax
    probs = forward(h1,params,'output',softmax)
    # make sure you understand these values!
    # positive, ~1, ~1, (40,4)
    print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

    # Q 2.2.3
    # implement compute_loss_and_acc
    loss, acc = compute_loss_and_acc(y, probs)
    # should be around -np.log(0.25)*40 [~55] or higher, and 0.25
    # if it is not, check softmax!
    print("{}, {:.2f} loss and accuracy".format(loss,acc))

    #? TA comments
    # here we cheat for you
    # the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]

    #? My comments
    # derivative in front of (upstream of softmax)
    delta1 = probs - y

    #? TA comments
    # we already did derivative through softmax
    # so we pass in a linear_deriv, which is just a vector of ones to make this a no-op

    #? Reason behind using linear_deriv (np.ones) as softmax derivative #
    # In pytorch the cross entropy loss includes the softmax on final layer + loss function
    # Therefore in pytorch's case, if we calculate derivative of cross-entropy loss, 
    #  we automatically have calculated the derivate through softmax as well

    #? My comments
    # basically the derivative w.r.t the softmax function should only return an array of np.ones
    # of the same shape as the post_activation shape of softmax( 40,4) -> this is done by linear_deriv

    # the backwards function does the following in order
    # 1. Finds derivative through activation function (here linear_deriv)
    # 2. Find derivative to bias
    # 3. Finds derivative to W and X
    # Finally, it returns derivative on X as delta
    delta2 = backwards(delta1, params, 'output', linear_deriv)
    print("delta2 shape is", delta2.shape)

    delta3 = backwards(delta2,params,'layer1',sigmoid_deriv)
    print("delta3 shape is", delta3.shape)

    # W and b should match their gradients sizes
    print("name | grad_shape | weight shape")
    for k,v in sorted(list(params.items())):
        if 'grad' in k:
            name = k.split('_')[1]
            print(name,v.shape, params[name].shape)

    #________________________________________________________________________________________#

    # Q 2.4
    batches = get_random_batches(x,y,5)
    # print batch sizes
    print([_[0].shape[0] for _ in batches])

    ######### WRITE A TRAINING LOOP HERE #################
    train_loop(batches)


    # Q 2.5 should be implemented in this file
    # you can do this before or after training the network. 

    # compute gradients using forward and backward
    h1 = forward(x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(y, probs)
    delta1 = probs - y
    delta2 = backwards(delta1,params,'output',linear_deriv)
    backwards(delta2,params,'layer1',sigmoid_deriv)

    # save the old params
    import copy
    params_orig = copy.deepcopy(params)

    # compute gradients using finite difference

    eps = 1e-6 # epsilon value

    # for k,v in params.items():
    #     if '_' in k: 
    #         continue
        # we have a real parameter!
        # for each value inside the parameter
        #   add epsilon
        #   run the network
        #   get the loss
        #   subtract 2*epsilon
        #   run the network
        #   get the loss
        #   restore the original parameter value
        #   compute derivative with central diffs

    forward_loss, forward_params = grad_check_loop(params, eps, x, y)
    backward_loss, backward_params = grad_check_loop(forward_params, -2*eps, x, y)

    numerical_grad = (forward_loss-backward_loss)/(2*eps)
    print("numerical grad is", numerical_grad)
    params = backward_params
    
    total_error = 0
    for k in params.keys():
        if 'grad_' in k:
            # relative error
            err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
            err = err.sum()
            print('{} {:.2e}'.format(k, err))
            total_error += err
    # should be less than 1e-4
    print('total {:.2e}'.format(total_error))



def grad_check_loop(params, eps, x, y):
    for k,v in params.items():
        if '_' in k: 
            continue
        else:
            params[k] += eps

    h1 = forward(x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(y, probs)
    return loss, params



def train_loop(batches):
    # init weights
    # parameters in a dictionary
    params = {}

    # initialize a layer
    initialize_weights(2,25,params,'layer1')
    initialize_weights(25,4,params,'output')

    max_iters = 500
    learning_rate = 0.5e-3
    # with default settings, you should get loss < 35 and accuracy > 75%
    for itr in range(max_iters):
        total_loss = 0
        avg_acc = 0
        for xb,yb in batches:

            # implement forward
            h1 = forward(xb,params,'layer1', sigmoid)

            # implement softmax
            probs = forward(h1,params,'output',softmax)
            # print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

            # loss
            loss, acc = compute_loss_and_acc(yb, probs)
            total_loss += loss
            avg_acc += acc
            # print("{}, {:.2f}".format(loss,acc))

            # backward
            delta1 = probs - yb

            delta2 = backwards(delta1, params, 'output', linear_deriv)

            delta3 = backwards(delta2,params,'layer1',sigmoid_deriv)


            # apply gradient 
            # gradients should be summed over batch samples
            for k,v in sorted(list(params.items())):
                if 'grad' in k:
                    name = k.split('_')[1]
                    # print(name,v.shape, params[name].shape)
                    params[name] -= learning_rate * v

        avg_acc = avg_acc/len(batches)    
        if itr % 100 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


if __name__ == '__main__':
    main()