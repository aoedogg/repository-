from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras  # tf.keras
import matplotlib
import matplotlib.pyplot as plt

housing = fetch_california_housing()

print(type(housing.data))
print(housing.data.shape)
print(housing.target.shape)

# Split data into training / validation / testing sets
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.3, random_state=42)
X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
print("Shape of training data: ", X_train.shape)
print("Shape of testing data: " ,X_test.shape)
print("Shape of validation data: " ,X_validate.shape)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled.shape)
X_valid_scaled = scaler.transform(X_validate)
X_test_scaled = scaler.transform(X_test)

# Train network on normalized data
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="relu", input_shape=X_train_scaled.shape[1:]))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(1e-3))
history = model.fit(X_train_scaled, y_train, epochs=25,
                    validation_data=(X_valid_scaled, y_validate))

print(model.evaluate(x=X_test_scaled, y=y_test))
fig = plt.figure(1)
plt.scatter(model.predict(X_test_scaled), y_test)
plt.plot(y_test, y_test, 'k')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


