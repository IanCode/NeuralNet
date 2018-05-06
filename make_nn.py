'''
References:
"Tensorflow Tutorial"
http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"Keras Tutorial: The Ultimate Beginner's Guide to Deep Learning in Python"
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
https://arxiv.org/abs/1409.1556
"Stanford University: Introduction to Convolutional Neural Networks for Visual Recognition"
http://cs231n.stanford.edu/slides/2017/

Data:
https://www.kaggle.com/c/dogs-vs-cats/data

Ian White 2018
'''

import sys
import numpy
import theano
import glob
import os
import cv2
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Activation, Dropout, ZeroPadding2D
from keras import backend

def main():
    print(numpy.__version__)
    print(theano.__version__)
    width, height = 100, 100
    x_train = []
    y_train = []
    #take training directory  and output file name as command line arguments
    cats_and_dogs = sys.argv[1]
    network_name = sys.argv[2]

    dir_path = os.path.dirname(os.path.realpath(__file__))

    print(dir_path)
    for filename in os.listdir(cats_and_dogs):
        if filename[0] == 'd':
            y_train.append((0,1))
            path = os.path.join(dir_path, cats_and_dogs, filename)
            img = cv2.imread(path)
            '''arr = np.array(img)
            if len(arr) == 2:
                np.append(arr, 3)'''
            x_train.append(img)
        elif filename[0] == 'c':
            y_train.append((1,0))
            path = os.path.join(dir_path, cats_and_dogs, filename)
            img = cv2.imread(path)
            '''arr = np.array(img)
            if len(arr) == 2:
                np.append(arr, 3)'''
            x_train.append(img)

    training_images = np.array(x_train)
    training_labels = np.array(y_train)


    num_train_samples = 25000
    epochs = 50
    batch_size = height+width

    model = Sequential()
    #generally, start with the lower level features (2x2 filter) and move to higher level features (5x5 filter)
    #from 64 to 32, I use a higher stride to downsample instead of using Max Pooling
        #In my opinion, this gives more precision than just taking a max from a region
        #while having a similar effect to a pooling layer
    #The results showed the loss function spiking down at the beginning, plateauing around
    #6.9 for a few epochs and then another steady drop until it began to approach 0 for the last 10-15 epochs
    #so somewhat of a downward exponential curve, could probably get away with 30 or so epochs
    #.9910 final training accuracy

    model.add(Conv2D(64, (2, 2), input_shape=(width, height, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), strides = 2))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Flatten before fully connected layers
    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(
        training_images,
        training_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose = 1)

    model.save(network_name)

if __name__ == "__main__":
    main()
