import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from keras.callbacks import EarlyStopping

df = pd.read_excel('clean_flats.xlsx')
features = ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square']

X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=42)


model = Sequential()

model.add(Dense(64, activation='sigmoid', input_dim=df[features].shape[1]))

model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))

model.add(Dense(1))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_mae', patience=10, mode='min', restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[early_stopping], validation_data=(X_test, y_test))
y_pred = model.predict(X_test)


print('оцінка якості моделі')
print("MAE (Mean Absolute Error): " + str(mean_absolute_error(y_test, y_pred)))
print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print("MAPE (Mean Absolute Percentage Error): " + str(mean_absolute_percentage_error(y_test, y_pred)))
print("r2 score: ", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--')
plt.xlabel("Фактичні значення")
plt.ylabel("Спрогнозовані значення")
plt.title("Спрогнозовані значення vs. Фактичні значення")

plt.show()

# print('оцінка точності')
# train_accuracy_score = model.score(X_train, y_train)
# print(train_accuracy_score)
#
# test_accuracy_score = model.score(X_test, y_test)
# print(test_accuracy_score)
