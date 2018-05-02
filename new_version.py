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

'''
import sys
import numpy
import theano
import glob
import os
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Activation, Dropout, ZeroPadding2D
from keras import backend

def main():
    print(numpy.__version__)
    print(theano.__version__)

    width, height = 100, 100

    #take training directory as a command line argument
    cats_and_dogs = sys.argv[1]

    '''dir_path = os.path.dirname(os.path.realpath(__file__))
    for filename in os.listdir('cats-and-dogs'):
    	if filename[0] == 'd':
    		newpath = dir_path+"\\dogs\\"+filename
    		oldpath = dir_path+path+ filename
    		print("Oldpath: " + oldpath)
    		print("Newpath: " + newpath)
    		os.rename(oldpath, newpath)
    	elif filename[0] == 'c':
    		newpath = dir_path+"\\cats\\"+filename
    		oldpath = dir_path+path+filename
    		print("Oldpath: " + oldpath)
    		print("Newpath: " + newpath)
    		os.rename(oldpath, newpath)'''






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
    #model.add(ZeroPadding2D((1,1), input_shape=(width, height, 3)))
    model.add(Conv2D(32, (3, 3), activation='elu',  input_shape=(width, height, 3)))
    #model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='elu'))
    #model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    '''model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))'''

    #Flatten before fully connected layers
    model.add(Flatten())

    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.5))

    '''model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2944, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='sigmoid'))
    model.add(Dropout(0.5))'''

    #it appears that sigmoid is more effective for two-class logistic regression
    model.add(Dense(1, activation='sigmoid'))

    #model.add(Dense(1, activation='softmax'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    trainImgGen = ImageDataGenerator(rescale=1. / 100)

    train_generator = trainImgGen.flow_from_directory(
        cats_and_dogs,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)

    model.save("ian.dnn")

if __name__ == "__main__":
    main()
