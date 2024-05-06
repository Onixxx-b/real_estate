import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error, \
    make_scorer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

df = pd.read_excel('../data/clean_flats_encoding_minus_percent.xlsx')
features = list(df.columns.drop('Price').drop('id'))
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=42)

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
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)

print('оцінка якості моделі')
print("MAE (Mean Absolute Error): " + str(mean_absolute_error(y_test, y_pred)))
print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print("MAPE (Mean Absolute Percentage Error): " + str(mean_absolute_percentage_error(y_test, y_pred)))
print("r2 score: ", r2_score(y_test, y_pred))


def custom_scoring_function(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)


perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42,
                                         scoring=make_scorer(custom_scoring_function))
mean_importance = perm_importance.importances_mean
feature_names = X_train.columns

sorted_indices = np.argsort(-mean_importance)
for i in sorted_indices:
    print(f"Feature '{feature_names[i]}', Importance: {mean_importance[i]:.4f}")

plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--')
plt.xlabel("Фактичні значення")
plt.ylabel("Спрогнозовані значення")
plt.title("Спрогнозовані значення vs. Фактичні значення")

plt.show()
