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
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Activation, Dropout
from keras import backend

def main():
    print(numpy.__version__)
    print(theano.__version__)

    # dimensions of our images.
    img_width, img_height = 100, 100

    #take training directory as a command line argument
    train_data_dir = sys.argv[1]
    validation_data_dir = 'validation'
    nb_train_samples = 12500
    nb_validation_samples = 5000
    epochs = 50
    batch_size = 16

    if backend.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    #output of convolutional layer: 
    # (input_size - filter_size + 2*pad_size)/stride + 1
    # (100 - 3 + 2(1))/2 + 1
    # (97 + 2)/2 + 1
    # (99/2) + 1
    model = Sequential()
    #start by making 3x2x2 convolutional layer
    model.add(ZeroPadding2D((1,1), input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    #model.add(Activation('elu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    #model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    #model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    #model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    #model.add(Activation('elu'))


    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Flatten before fully connected layers
    model.add(Flatten())

    model.add(Dense(256, activation='elu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    '''# this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)'''

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    '''validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')'''

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)
        #validation_data=validation_generator,
        #validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')
    model.save("ian.dnn")

if __name__ == "__main__":
    main()
