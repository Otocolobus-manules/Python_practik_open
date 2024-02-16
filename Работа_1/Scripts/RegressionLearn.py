from Scripts.SplitData import split_data
from sklearn.linear_model import QuantileRegressor


def regression_learning(data, sign) -> tuple:
    x_train, y_train, x_test, y_test = split_data(data=data, sign=sign)

    quantile_regressor = QuantileRegressor(quantile=0.53, alpha=0)
    quantile_regressor.fit(x_train, y_train)
    predictions = quantile_regressor.predict(x_test)
    return predictions, y_test
