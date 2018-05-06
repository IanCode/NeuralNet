# NeuralNet
Image classifier written in python

## Implementation Details
### Layers
Conv2d 64, filter of 3 x 3, stride of 1
output_size = (64 - 3)/1 + 1
output_size = 62
Conv2d 64, filter of 2 x 2, stride of 2
output_size = (N - F)/stride + 1
output_size = (64 - 2)/2 + 1
output_size = 62/2 + 1
output_size = 32

### General Layout Notes
- Fully Connected vs Convolutional
	- FC: each neuron looks at entire input region
	- Conv: each neuron looks at output section from filter
- Pooling Layer
	- makes representations smaller and more manageable
	- downsamples input volume, but doesn't change depth
	- Max Pooling:
		- Pooling layer with filter size and stride
		- instead of doing dot products, just take the max value of that region
		- in pooling layers, it is more common to have the stride so there is no overlap (multiple of same max for regions)
		- recently, people have used stride more to do the downsampling
		- Common Settings:
			- F = 2, S = 2
			- F = 3, S = 2
	- How much pooling is enough?
		- don't pool too much
		- trade off
- Common Format:
	- [(CONV-RELU)\*N - POOL?]\*M-(FC-RELU)\*K,SOFTMAX
		- conv-relu layer for some N number of steps followed by a pooling layer
		- that whole thing repeated some M number of times
		- followed by fully connected RELU layers some K number of times 
		- where N is usually up to 5, M is large, 0 <= K <= 2
		- softmax at the end for class scores
		- few fully connected layers at the end

## Recommended Packages
- Anaconda: https://www.anaconda.com/download/
- matplotlib

## Required Packages
- scipy with numpy
- keras
- tensorflow

## Notes on Neural Nets
### Activation Functions
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

### Optimizations

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
- convoluting by `n x n` kernel results in a size reduction of `n - 1`
- *eg.*
	- `4 x 4` matrix
	- `2 x 2` kernel (convolution matrix)
	- results in `3 x 3` matrix


##### Convolutional Network Layers
- create whole layers of neural network based on matrix convolution
- kernels of these layers are analagous to neurons in a neural network
- then feed the output of that into traditional layers of a fully connected neural net

## References
`Building powerful image classification models using very little data`
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
`Tensorflow Tutorial`
http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
`VGG16 model for Keras`
https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
`Very Deep Convolutional Networks for Large-Scale Image Recognition`
https://arxiv.org/abs/1409.1556

## Data
https://www.kaggle.com/c/dogs-vs-cats/data
