# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:11:35 2019

@author: Andre
"""

from keras.datasets import cifar10
import keras.utils as utils
import numpy as np
from keras.models import load_model
import matplotlib


#DADOS:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

respostas = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

imagem = matplotlib.pyplot.imshow(np.asarray(x_test[6503]))

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_train = x_train / 255.0
x_test = x_test.astype('float32')
x_test = x_test / 255.0

#IMPORTAÇÃO DO MODELO TREINADO:

rede = load_model(filepath='Cifar_10.h5')

#PREDIÇÃO DA REDE:

#rede.evaluate(x_test, y_test)

predição = rede.predict(np.asarray([x_test[6503]]))

max_index = np.argmax(predição[0])

print(respostas[max_index])

