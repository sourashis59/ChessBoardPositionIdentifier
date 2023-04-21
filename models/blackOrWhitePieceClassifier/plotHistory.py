import time
import tensorflow as tf
import os 
import cv2
# import imghdr
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall , BinaryAccuracy, SparseCategoricalAccuracy, Accuracy
import pickle



# history = pickle.load(open('models/pieceTypeClassifier/history.h5', 'rb'))
history = np.load('models/blackOrWhitePieceClassifier/history.npy', allow_pickle='TRUE').item()
# print(history)


def plot_loss_acc(history):
    '''plots the training and validation loss and sparse_categorical_accuracy from a history object'''
    acc = history['sparse_categorical_accuracy']
    val_acc = history['val_sparse_categorical_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training sparse_categorical_accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation sparse_categorical_accuracy')
    plt.title("Training and Validation sparse_categorical_accuracy")

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()




# *plot training results
plot_loss_acc(history) 