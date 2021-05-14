from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as mat
import tensorflow as tf
from cv2 import cv2


mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=tf.nn.relu,padding="same",input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation=tf.nn.relu,padding="same"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding="same"))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=tf.nn.relu,padding="same"))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation=tf.nn.relu,padding="same"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding="same"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
model.fit(x_train,y_train,epochs=2, validation_data=(x_test,y_test))

accuracy,loss = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('C:/repository/OCR/modelos/digits.model')
