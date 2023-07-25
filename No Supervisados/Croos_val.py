import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (cross_val_score, KFold)

dataset = pd.read_csv('./data/felicidad.csv')
X = dataset.drop(['country', 'score'], axis=1)
y = dataset['score']
print(dataset.shape)

model = DecisionTreeRegressor()

score = cross_val_score(model, X,y, cv=3, 
scoring='neg_mean_squared_error') # con cv podemos controlar el numOfFolds
print(score)

print(np.abs(np.mean(score)))

kf = KFold(n_splits=3, shuffle=True, random_state=42)
mse_values = []
for train, test in kf.split(dataset):
    print(train)
    print(test)

    X_train = X.iloc[train]
    y_train = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

    model = DecisionTreeRegressor().fit(X_train, y_train)
    predict = model.predict(X_test)
    mse_values.append(mean_squared_error(y_test, predict))
    
    print("Los tres MSE fueron: ", mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))



