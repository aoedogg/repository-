import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras  # tf.keras
import seaborn as sns
from sklearn.decomposition import PCA

data = pd.read_csv('data.csv')
print(data.shape)
print(data.head())

# Convert categorical data into binary data for classification
data['diagnosis'] = pd.Categorical(data['diagnosis'])
data['diagnosis'] = data['diagnosis'].cat.codes
data.drop(columns=['Unnamed: 32'], inplace = True)
print(data.columns)
# Plot data
jj = 1
for column in data:
    plt.hist(data[column])
    plt.subplot(8,4,jj)
    jj = jj + 1
#plt.show()

# Split into X, y
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']
print(X.head(10))

plt.subplots(figsize=(20, 15))
sns.heatmap(X.corr(), annot = True)
#plt.show()
# Drop columns based on correlation analysis
X = data.drop(columns=['id', 'diagnosis','perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'texture_worst','perimeter_se', 'area_se'])


#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

#PCA
numc = 10
pca = PCA(n_components=numc)
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
print(X_train_scaled_pca.shape)
print(pca.explained_variance_ratio_)


X_test_scaled = scaler.transform(X_test)
X_test_scaled_pca = pca.transform(X_test_scaled)

# Testing / Validation split
X_validate_scaled_pca, X_test_scaled_pca, y_validate, y_test = train_test_split(X_test_scaled_pca, y_test, test_size=0.5, random_state=42)
print("Shape of training data: ", X_train_scaled_pca.shape)
print("Shape of testing data: " ,X_test_scaled_pca.shape)
print("Shape of validation data: " ,X_validate_scaled_pca.shape)


# Train logistic regression
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, input_dim=numc, activation='relu'))
model.add(keras.layers.Dense(15, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(lr=5e-3),
              metrics=["binary_accuracy"])

history = model.fit(X_train_scaled_pca, y_train, epochs=100,
                    validation_data=(X_validate_scaled_pca, y_validate))


print(model.evaluate(X_test_scaled_pca, y_test))
