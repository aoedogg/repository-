import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras  # tf.keras
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
import sys

# Our top priority in this business problem is to identify customers who are getting churned.
# Even if we predict non-churning customers as churned, it won't harm our business. But predicting churning customers as Non-churning will do. So recall (TP/TP+FN) need to be higher.

data = pd.read_csv('BankChurners.csv')
data.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], inplace = True)
data.drop(columns=['CLIENTNUM'], inplace = True)

print(data.shape)
# One hot encoding for multi-category features
Education_Level_ohe = pd.get_dummies(data['Education_Level'], prefix='Education')
data.join(Education_Level_ohe)
Income_Category_ohe = pd.get_dummies(data['Income_Category'], prefix='Income')
data.join(Income_Category_ohe)
Card_Category_ohe = pd.get_dummies(data['Card_Category'], prefix='Card_Category')
data.join(Card_Category_ohe)
Marital_Status_ohe = pd.get_dummies(data['Marital_Status'], prefix='Marital_Status')
data.join(Marital_Status_ohe)
data.drop(columns=['Education_Level','Income_Category','Card_Category', 'Marital_Status'], inplace = True)

# Here  binary categorical variables are converted into ints.
cat_ints = ['Attrition_Flag', 'Gender']
for columns in cat_ints:
    data[columns] = pd.Categorical(data[columns])
    data[columns] = data[columns].cat.codes

print(data.columns)
print(data.head(100))

# Split into X, y
X = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testing / Validation split
X_validate_scaled, X_test_scaled, y_validate, y_test = train_test_split(X_test_scaled, y_test, test_size=0.5, random_state=42)
print("Shape of training data: ", X_train_scaled.shape)
print("Shape of testing data: " ,X_test_scaled.shape)
print("Shape of validation data: " ,X_validate_scaled.shape)

# number of input features
num_feat = X_train_scaled.shape[1]

# bundle model so we can plot confusion matrix with SKlearn
def wrapped_model(input_dim=num_feat):
    # Train logistic regression
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(30, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dense(15, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(lr=5e-3),
                  metrics=["binary_accuracy"])
    return model

keras_clf = keras.wrappers.scikit_learn.KerasClassifier(wrapped_model)
keras_clf.fit(X_train_scaled, y_train, epochs=100,
              validation_data=(X_validate_scaled, y_validate))

y_pred = (keras_clf.predict(X_test_scaled))
print(confusion_matrix(y_test, y_pred))
print(recall_score(1 - y_test, 1 -  y_pred))
