import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras  # tf.keras
import time
from sklearn.preprocessing import StandardScaler

# Get dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Print example from training set
plt.imshow(X_train[0], cmap="binary")
plt.text(0.5, 0.5, y_train[0])
plt.show()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))

print(model.evaluate(X_test, y_test))

# Normalizing training data for better fit

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32).reshape(-1, 28 * 28)).reshape(-1, 28, 28) # fit and scale
print(X_train_scaled.shape)
X_valid_scaled = scaler.transform(X_valid.astype(np.float32).reshape(-1, 28 * 28)).reshape(-1, 28, 28) # scale
X_test_scaled = scaler.transform(X_test.astype(np.float32).reshape(-1, 28 * 28)).reshape(-1, 28, 28) # scale

# Train on normalized data
model2 = tf.keras.models.clone_model(model)
model2.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history2 = model2.fit(X_train_scaled, y_train, epochs=10,
                    validation_data=(X_valid_scaled, y_valid))

print(model2.evaluate(X_test_scaled, y_test))

# Training deep NN on example
model3 = keras.models.Sequential()
model3.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model3.add(keras.layers.Dense(100))
    model3.add(keras.layers.BatchNormalization())
    model3.add(keras.layers.Activation("relu"))
    #model3.add(keras.layers.Dropout(rate = 0.5))

model3.add(keras.layers.Dense(10, activation="softmax"))
model3.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(1e-3),
              metrics=["accuracy"])
history3 = model3.fit(X_train_scaled, y_train, epochs=20,
                    validation_data=(X_valid_scaled, y_valid))


print(model3.evaluate(X_test_scaled, y_test))