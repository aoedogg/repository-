import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras  # tf.keras
import seaborn as sns
from sklearn.decomposition import PCA

data = pd.read_csv('BankChurners.csv')
print(data.shape)
print(data.head())