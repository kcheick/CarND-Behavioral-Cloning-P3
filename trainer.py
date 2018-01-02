import csv 
import cv2
import numpy as np

drivingLogPath = './windows_sim/recording/driving_log.csv'
images = []
measurements = []
# csv format 
# center image, Left Image, Right Imge, Steering Angle, Throttle, Break, Speed
with open(drivingLogPath) as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        center_image = line[0]
        images.append(cv2.imread(center_image))
        measurements.append(float(line[3]))


X_train = np.array(images)
y_train = np.array(measurements)

# building a regression model for the sterring value 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
import keras


def simpleDnn(X_train, y_train):
    model = Sequential()
    # normalizing and mean centering
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X_train[0].shape))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss=keras.losses.mean_squared_error, optimizer= keras.optimizers.adam())
    model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, epochs=5)

    return model
    model.save('model.h5')

def lenet(X_train, y_train):
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = X_train[0].shape

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])






simple = simpleDnn()
simple.save('simple.h5')

