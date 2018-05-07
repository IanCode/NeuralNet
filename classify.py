import sys
import numpy
import theano
import glob
import os
import cv2
from keras import models
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Activation, Dropout, ZeroPadding2D
from keras import backend


def main():
    model_name = sys.argv[1]
    batch_size = len(sys.argv) - 1
    x_train = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, model_name)
    model = models.load_model(model_path)
    image_names = []
    #read all the image arguments
    for i in range(2, len(sys.argv)):
        filename = sys.argv[i]
        image_names.append(filename)
        #y_train.append((0,1))
        path = os.path.join(dir_path, filename)
        img = cv2.imread(path)
        x_train.append(img)

    training_images = np.array(x_train)

    predictions = model.predict(training_images, batch_size)

    index = 0
    for prediction in predictions:
        if prediction[0] > prediction[1]:
            print(image_names[index]+ " is a cat.")
        else:
            print(image_names[index]+ " is a dog.")
        index = index + 1

if __name__ == "__main__":
    main()
