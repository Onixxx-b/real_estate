import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

df = pd.read_excel('clean_flats.xlsx')
features = ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=42)


# Функция для создания модели
def create_model(optimizer='adam', activation='relu', units=64, dropout_rate=0.2):
    model = Sequential([
        Dense(units, activation=activation, input_shape=(df[features].shape[1],)),
        Dropout(dropout_rate),
        Dense(units, activation=activation, input_shape=(df[features].shape[1],)),
        Dropout(dropout_rate),
        Dense(units, activation=activation, input_shape=(df[features].shape[1],)),
        Dropout(dropout_rate),
        Dense(units, activation=activation),
        Dropout(dropout_rate),
        Dense(units, activation=activation),
        Dropout(dropout_rate),
        Dense(units, activation=activation),
        Dropout(dropout_rate),
        Dense(units, activation=activation),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Оборачиваем модель в KerasRegressor
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Задаем параметры для перебора
param_grid = {
    'units': [64, 128, 256],
    'dropout_rate': [0.1, 0.2],
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'sigmoid']
}

# Инициализируем GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)

# Выполняем поиск лучших параметров
grid_result = grid_search.fit(X_train, y_train)

# Выводим лучшие параметры и результаты
print("Лучшие параметры: ", grid_result.best_params_)
print("Лучшее значение MSE: ", -grid_result.best_score_)
