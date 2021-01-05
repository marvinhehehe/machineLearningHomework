import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from numpy import *
import csv
import os
import cv2
from skimage.filters import threshold_otsu
from skimage import feature
from random import randint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

testSequence = []
trainLabel = ndarray(shape=[28709, 7])
trainData = ndarray(shape=[28709, 48, 48])

def getData():
    number = 0
    for filename in os.listdir(r"./train/angry"):
        position = "train/angry/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [1, 0, 0, 0, 0, 0, 0]
        trainLabel[number] = label
        number = number + 1
    for filename in os.listdir(r"./train/disgust"):
        position = "train/disgust/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [0, 1, 0, 0, 0, 0, 0]
        trainLabel[number] = label
        number = number + 1
    for filename in os.listdir(r"./train/fear"):
        position = "train/fear/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [0, 0, 1, 0, 0, 0, 0]
        trainLabel[number] = label
        number = number + 1
    for filename in os.listdir(r"./train/happy"):
        position = "train/happy/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [0, 0, 0, 1, 0, 0, 0]
        trainLabel[number] = label
        number = number + 1
    for filename in os.listdir(r"./train/neutral"):
        position = "train/neutral/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [0, 0, 0, 0, 1, 0, 0]
        trainLabel[number] = label
        number = number + 1
    for filename in os.listdir(r"./train/sad"):
        position = "train/sad/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [0, 0, 0, 0, 0, 1, 0]
        trainLabel[number] = label
        number = number + 1
    for filename in os.listdir(r"./train/surprise"):
        position = "train/surprise/" + filename
        img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
        trainData[number] = img1 / 255.0
        label = [0, 0, 0, 0, 0, 0, 1]
        trainLabel[number] = label
        number = number + 1


if __name__ == '__main__':
    num_labels = 7
    batch_size = 128
    epochs = 2
    width, height = 48, 48

    getData()

    trainData = array(trainData)
    X_train = array(trainData)
    y_train = array(trainLabel)
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

    model = tf.keras.Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(width, height, 1),
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))


    model.add(Flatten())

    model.add(Dense(1024, activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random() * 100))))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.0001, patience=10, verbose=1,
                                   min_lr=0.000001)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs, validation_data=(X_val, y_val),
                        callbacks=[lr_reducer])

    model.save('models/fer_model.h5')

    # load and finetune the pre-trained model
    model2 = load_model('models/fer_model.h5')

    model2.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00003, decay=1e-6, momentum=0.9, nesterov=True),
                   loss=tf.keras.losses.categorical_crossentropy,
                   metrics=['accuracy'])

    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6, verbose=1, min_delta=0.00001,
                                   min_lr=0.0000001)

    history2 = model2.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(X_val, y_val),
                          callbacks=[lr_reducer])

    model2.save('models/fer_model_finetuned.h5')
