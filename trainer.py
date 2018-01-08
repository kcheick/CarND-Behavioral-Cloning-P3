import csv 
import cv2
import numpy as np


# building a regression model for the sterring value 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, convolutional
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras import backend as K
from pyxtension.streams import stream as pyStream





def readData(drivingLogPath='./windows_sim/recording/driving_log.csv', augment=(lambda x,y: ([x],[y]) )):
    images = []
    measurements = []
    # csv format 
    # center image, Left Image, Right Imge, Steering Angle, Throttle, Break, Speed
    with open(drivingLogPath) as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            center_image = line[0]
            image = cv2.imread(center_image)
            measurement = float(line[3])
            augmented_images, augemented_measurement = augment(image, measurement)
            images.extend(augmented_images)
            measurements.extend(augemented_measurement)

def readDataGen(drivingLogPath='./windows_sim/recording/driving_log.csv'):
    # csv format 
    # center image, Left Image, Right Imge, Steering Angle, Throttle, Break, Speed
    with open(drivingLogPath) as csvFile:
        reader = csv.reader(csvFile)
        correction = 0.2
        for line in reader:
            center_image    = cv2.imread(line[0]) 
            left_image      = cv2.imread(line[1]) 
            right_image     = cv2.imread(line[2]) 
            measurement     = float(line[3])
            yield [
                    (center_image, measurement),
                    (left_image, measurement + correction),
                    (right_image, measurement - correction)
                  ]
            

def augment_data(image, measurement):
    return [(image, measurement), (np.fliplr(image), - measurement)]


rd = pyStream(readDataGen()).flatMap().map(lambda data: augment_data(data[0], data[1])).flatMap()
print(type(rd))
xt, yt = zip(*rd)
X_train, y_train =  (np.array((xt)), np.array((yt)))
print(type(X_train[0]))
print(X_train[0].shape)
# print(y_train[0])
# exit()

# X_train, y_train = 


# X_train, y_train = readData(augment=lambda image, meas: ([image, np.fliplr(image)], [meas, -meas]))
#print(y_train)
#print(y_train[1])



def simpleDnn(X_train, y_train):
    model = Sequential()
    # normalizing and mean centering
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train[0].shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss=keras.losses.mean_squared_error, optimizer= keras.optimizers.adam())
    model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, epochs=5)

    return model
    model.save('model.h5')



def lenet(X_train, y_train):
    batch_size = 32
    num_classes = 10
    epochs = 12

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train[0].shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(
        loss=keras.losses.mean_squared_error, 
        optimizer= keras.optimizers.adam())
    model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, epochs=epochs, batch_size=batch_size)
    return model


def nvidia(X_train, y_train):
    batch_size = 32
    num_classes = 10
    epochs = 5

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train[0].shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24, kernel_size=(5, 5), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, kernel_size=(5, 5), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, kernel_size=(5, 5), subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(
        loss=keras.losses.mean_squared_error, 
        optimizer= keras.optimizers.adam())
    model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, epochs=epochs, batch_size=batch_size)
    return model 

nvidia(X_train, y_train).save("nvida.h5")
#lenet(X_train, y_train)
#simpleDnn(X_train, y_train)



#simple = simpleDnn()
#simple.save('simple.h5')


