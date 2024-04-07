from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_excel('clean_flats_02_2024.xlsx')
features = ['Region', 'Room count', 'Total Square', 'Floor', 'Subway', 'Max Floor', 'Living Square', 'Kitchen Square']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Price'], test_size=0.2, random_state=42)

param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=XGBRegressor(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(best_params)
