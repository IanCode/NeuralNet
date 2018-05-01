# NeuralNet
Image classifier written in python

## Recommended Packages
- Anaconda: https://www.anaconda.com/download/
- matplotlib

## Required Packages
- scipy with numpy
- keras
- tensorflow

### Notes on Activation Functions
#### RELU
- Quick to calculate
- 'sigmoid that doesn't saturate on the high end'
- trained more easily, even in a deep network (viable deep networks)
- can have dying relu problem
- can get locked into returning 0 always

#### Leaky RELU
- derivative is never 0
- may still be adjsuting very slowly

#### Softplus
- behaves like a RELU

#### ELU
- continuous and differentiable at 0
- helps mitigate vanishing gradients

#### AdaGrad
- different learning rate for every weight
- scaled automatically
- can reduce learning rates too much sometimes

#### RMSProp
- similar to adagrad but it limits the amount of decay done each step

#### Adam
- currently favored methodology
- algorithm that combines RMSProp style learning decay and momentum

#### Dropout  
- every epoch of training, remove some of the neurons
- randomly choose neurons to drop
- after training, put back
- builds redundancy into the system
- doesn't rely on any one neuron
- in some ways, we are actually training a massive ensemble of networks
- most libraries support Dropout
- increase accuracy with very little cost, can turn on by adding an argument to one of the functions

### Convolutional Neural Networks
- Uses multidimensional representations of matrices called tensors
##### Matrix Convolution
- flip second matrix around
- take dot product of two matrices
- but actually wait no the second is gonna be smaller than the first
- this matrix is called the kernel
- slide kernel over first matrix, compare in many different places
- at each position, take dot product, and put all those dot products into one matrix
- convolution matrix is usually an odd sized square

##### Convolutional Network Layers
- create whole layers of neural network based on matrix convolution
- kernels of these layers are analagous to neurons in a neural network
- then feed the output of that into traditional layers of a fully connected neural net
- 
