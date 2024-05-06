import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_excel('../data/clean_flats_encoding_minus_percent.xlsx')
features = list(df.columns.drop('Price').drop('id'))
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100,
                     learning_rate=0.1,
                     subsample=0.7,
                     colsample_bytree=0.9,
                     alpha=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('оцінка якості моделі')
print("MAE (Mean Absolute Error): " + str(mean_absolute_error(y_test, y_pred)))
print("RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print("MAPE (Mean Absolute Percentage Error): " + str(mean_absolute_percentage_error(y_test, y_pred)))
print("r2 score: ", r2_score(y_test, y_pred))

# plt.scatter(y_test, y_pred, color='red')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--')
# plt.xlabel("Фактичні значення")
# plt.ylabel("Спрогнозовані значення")
# plt.title("Спрогнозовані значення vs. Фактичні значення")
#
# plt.show()

feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
sorted_importance = feature_importance.sort_values(ascending=False)

plt.bar(sorted_importance.index, sorted_importance)
plt.xlabel('Ознаки')
plt.ylabel('Значимість')
plt.title('Значимість ознак')
plt.xticks(rotation=45)
plt.show()
