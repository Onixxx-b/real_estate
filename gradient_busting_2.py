import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_excel('clean_flats_02_2024_2.xlsx')
features = ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=0)

model = XGBRegressor(n_estimators=100,
                     learning_rate=0.1,
                     subsample=0.7,
                     colsample_bytree=0.9,
                     alpha=0.5)
model.fit(X_train, y_train)
# eval_set = [(X_train, y_train), (X_test, y_test)]
# model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=10, verbose=True)
y_pred = model.predict(X_test)

# lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
#
# for learning_rate in lr_list:
#     gb_clf = XGBRegressor(n_estimators=100, learning_rate=learning_rate, max_depth=10,
#                           random_state=0)
#     gb_clf.fit(X_train, y_train)
#
#     print("Learning rate: ", learning_rate)
#     print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
#     print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

print('оцінка якості моделі')
print("MAE (Mean Absolute Error): " + str(mean_absolute_error(y_test, y_pred)))
print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print("MAPE (Mean Absolute Percentage Error): " + str(mean_absolute_percentage_error(y_test, y_pred)))
print("r2 score: ", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--')
plt.xlabel("Фактичні значення")
plt.ylabel("Спрогнозовані значення")
plt.title("Спрогнозовані значення vs. Фактичні значення")

plt.show()
