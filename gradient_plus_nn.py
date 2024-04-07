import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import xgboost as xgb
import pickle


df = pd.read_excel('clean_flats.xlsx')
features = ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square']

X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=0)


num_networks = 5
networks = []
for i in range(num_networks):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(df[features].shape[1],)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    networks.append(model)

for i, model in enumerate(networks):
    model.fit(X_train, y_train, epochs=10, batch_size=64)
    print(f"Network {i+1} trained.")
    model.save(f"saved/model_{i}.h5")

y_pred_networks = np.zeros((num_networks, len(X_test)))
for i, model in enumerate(networks):
    y_pred_networks[i] = model.predict(X_test).flatten()

boosting_model = xgb.XGBRegressor()
boosting_model.fit(y_pred_networks.T, y_test)
with open("saved/xgb_model.pkl", "wb") as f:
    pickle.dump(boosting_model, f)

y_pred_boosting = boosting_model.predict(y_pred_networks.T)

print('оцінка якості моделі')
print("MAE (Mean Absolute Error): " + str(mean_absolute_error(y_test, y_pred_boosting)))
print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred_boosting))))
print("MAPE (Mean Absolute Percentage Error): " + str(mean_absolute_percentage_error(y_test, y_pred_boosting)))
print("r2 score: ", r2_score(y_test, y_pred_boosting))


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_boosting, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--')
plt.xlabel("Фактичні значення")
plt.ylabel("Спрогнозовані значення")
plt.title("Спрогнозовані значення vs. Фактичні значення")

plt.show()



# Test
# feature1 = np.array([4, 2, 72.5, 10, 0, 16, 20.5, 30])  # 85000 -> 122699
# feature2 = np.array([1, 1, 33, 2, 0, 5, 18, 7])  # 43000 ->  36050
# feature3 = np.array([1, 2, 66.5, 16, 1, 18, 25, 19])  # 132500 ->  95825.13
# X_new = np.column_stack((feature3))
#
# y_pred_networks_new = np.zeros((num_networks, len(X_new)))
# for i, model in enumerate(networks):
#     y_pred_networks_new[i] = model.predict(X_new).flatten()
#
# y_pred_boosting_new = boosting_model.predict(y_pred_networks_new.T)
#
# print("Results for testing")
# print(y_pred_boosting_new)
