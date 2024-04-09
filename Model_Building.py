# building the model:
# since we do not have a lot of features, we will not do feature selection, and we wish to keep them.
# which means we will use RandomForestRegressor for linear regression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def model_build(X, y):
    #e Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

    #e Create a Gradient Boosted Regression Trees model
    model = RandomForestRegressor(n_estimators=1000, random_state=12345)
    #e Train the model on the training data
    model.fit(X_train, y_train)
    return model, X_test, y_test
def model_evaluate(model, X_test, y_test):
    #e Make predictions on the test data
    y_pred = model.predict(X_test)
    print(len(y_pred))
    #e Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error:{rmse}")
