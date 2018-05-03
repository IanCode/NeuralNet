'''
References:
"Building powerful image classification models using very little data"
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"Tensorflow Tutorial"
http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"VGG16 model for Keras"
https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
https://arxiv.org/abs/1409.1556

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
    #cats = []
    #dogs = []
    x_train = []
    y_train = []
    #take training directory as a command line argument
    cats_and_dogs = sys.argv[1]
    network_name = sys.argv[2]

    dir_path = os.path.dirname(os.path.realpath(__file__))

    print(dir_path)
    for filename in os.listdir(cats_and_dogs):
        print(os.path.join(dir_path, cats_and_dogs, filename))
        if filename[0] == 'd':
            y_train.append(0)
            path = os.path.join(dir_path, cats_and_dogs, filename)
            img = cv2.imread(path)
            arr = np.array(img)
            if len(arr) == 2:
                np.append(arr, 3)   
            x_train.append(arr)
        elif filename[0] == 'c':
            y_train.append(1)
            path = os.path.join(dir_path, cats_and_dogs, filename)
            img = cv2.imread(path)
            arr = np.array(img)
            if len(arr) == 2:
                np.append(arr, 3)
            print("doing it")
            print(arr.shape)
            x_train.append(arr)

    training_images = np.array(x_train)
    training_labels = np.array(y_train)


    nb_train_samples = 25000
    epochs = 50
    batch_size = 16

    #output of convolutional layer:
    # (input_size - filter_size + 2*pad_size)/stride + 1
    # (100 - 3 + 2(1))/2 + 1
    # (97 + 2)/2 + 1
    # (99/2) + 1

    model = Sequential()
    #start by making 3x2x2 convolutional layer
    model.add(Conv2D(32, (3, 3), activation='elu',  input_shape=(width, height, 3)))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Flatten before fully connected layers
    model.add(Flatten())

    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.5))

    #it appears that sigmoid is more effective than softmax
    #for two-class logistic regression
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    trainImgGen = ImageDataGenerator(rescale=1. / 100)

    train_generator = trainImgGen.flow_from_directory(
        cats_and_dogs,  # this is the target directory
        target_size=(height, width),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary') # since we use binary_crossentropy loss, we need binary labels

    train_generator = trainImgGen.flow(training_images, y=training_labels, batch_size=batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)

    model.save(network_name)

if __name__ == "__main__":
    main()
