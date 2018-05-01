# NeuralNet
Image classifier written in python

# Recommended Packages
Anaconda: https://www.anaconda.com/download/
matplotlib

## Required Packages:
scipy with numpy

## Notes on Activation Functions
#### RELU
Quick to calculate
'sigmoid that doesn't saturate on the high end'
trained more easily, even in a deep network (viable deep networks)
can have dying relu problem
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
