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
3. 