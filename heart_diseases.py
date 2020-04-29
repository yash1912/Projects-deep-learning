import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# PRE-PROCESSING DATA
data = pd.read_csv(r"C:\Users\Yash\Desktop\yash\heart.csv")
#print(data.info())
#data.hist()
#dataset = pd.get_dummies(data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
data_shuffle = data.sample(frac=1).reset_index(drop=True)
y = data_shuffle['target']
X = data_shuffle.drop(['target'], axis = 1)
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.15)

# NEURAL NETWORK
model = keras.Sequential([
    keras.layers.Dense(13,activation='relu',kernel_initializer='random_uniform'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train.values,y_train.values,epochs=70,validation_split=0.1)
model.evaluate(x_test.values,y_test.values)
