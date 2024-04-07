# Importing the necessary functionality
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D
from keras.layers import Flatten, MaxPooling2D
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Creating the model
DNN_Model = Sequential()

# Inputting the shape to the model
DNN_Model.add(Input(shape=(256, 256, 3)))

# Creating the deep neural network
DNN_Model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
DNN_Model.add(MaxPooling2D(2, 2))
DNN_Model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
DNN_Model.add(MaxPooling2D(2, 2))
DNN_Model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
DNN_Model.add(MaxPooling2D(2, 2))

# Creating the output layers
DNN_Model.add(Flatten())
DNN_Model.add(Dense(64, activation='relu'))
DNN_Model.add(Dense(10))

df = pd.read_excel('clean_flats_02_2024.xlsx')
features = ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square']

X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=0)
DNN_Model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
DNN_Model.fit(X_train, y_train, validation_data=(X_test, y_test))
y_pred = DNN_Model.predict(X_test)

print('оцінка якості моделі')
print("MAE (Mean Absolute Error): " + str(mean_absolute_error(y_test, y_pred)))
print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print("MAPE (Mean Absolute Percentage Error): " + str(mean_absolute_percentage_error(y_test, y_pred)))
print("r2 score: ", r2_score(y_test, y_pred))
