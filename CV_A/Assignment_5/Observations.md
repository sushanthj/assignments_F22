# Weight initialization

- Each neuron (say neuron bob) in a layer gets say 10 inputs 
  (from previous layer which has 10 neurons)
- Therefore neuron bob now needs 10 weights associated with all such inputs
- For every other neuron in this layer like bob, they will also have 10 inputs
- Therefore, we save the overall weights for each layer as a Nx10 matrix (where N=number of neurons
  in a layer)

Now, we'll initialize them by sampling random variables from a gaussian distribution:
(http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)


# Useful Links

## Weight initialization

1. https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
2. https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

## Pytorch

1. http://pytorch-sush.herokuapp.com/
2. https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader (both answers with TensorDataset and other one will definately be useful)
3. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
4. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
5. https://courses.cs.washington.edu/courses/cse446/19au/section9.html

### General Debugging
   1. It's easier to debug a network on CPU first, GPU obscures some debug messages
   2. In pytorch if you get 'Target 10 is out of bounds' understand that 'target' = labels
      And it's most likely out of bounds because the shape of final layer of your network is not
      matching the total number of class labels which pytorch has found (as in found by the 
      dataloader or defined by the user itself in say a train_labels.mat file)
   3. You can use x.shape even if x = torch.Tensor(), this will still output the shape


# Homework

1. The squeezenet gave higher accuracy but somehow took more number of epochs to train even though
   it was technically just fine-tuned
2. The learning rate was also quite high (should reduce to 0.5e-3 I think),     still final acc = 100%

3. squeezenet on 6.2 = 100%
4. sush_net on 6.2 = 28%