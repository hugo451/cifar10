# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:03:26 2019

@author: Andre
"""
#IMPORTAÇÔES:

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.datasets import cifar10
import keras.utils as utils
import numpy as np


#DADOS:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

respostas = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_train = x_train / 255.0
x_test = x_test.astype('float32')
x_test = x_test / 255.0

#DEFINIÇÃO DA REDE:

rede = Sequential()

#CAMADAS DE CONVOLUÇÃO E POOLINGS:

rede.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32,32,3), padding='same', kernel_constraint=maxnorm(3)))

rede.add(MaxPooling2D(pool_size=(2, 2)))

rede.add(Flatten())

#CAMADAS DENSAS E DROPOUTS:

rede.add(Dense(units=256, activation='relu'))

rede.add(Dropout(0.25))

rede.add(Dense(units=512, activation='relu'))

rede.add(Dropout(0.50))

rede.add(Dense(units=512, activation='relu'))

rede.add(Dropout(0.50))

rede.add(Dense(units=10, activation='softmax'))

#COMPILAÇÃO DA REDE:

rede.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#TREINAMENTO DA REDE:

rede.fit(x_train, y_train, batch_size=40, epochs=200)

#SALVAMENTO DOS PESOS APRENDIDOS:

rede.save(filepath='Cifar_10.h5')



