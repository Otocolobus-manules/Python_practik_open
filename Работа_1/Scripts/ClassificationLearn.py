from Scripts.SplitData import split_data
from sklearn.linear_model import Perceptron


def classification_learning(data, sign) -> tuple:
    x_train, y_train, x_test, y_test = split_data(data=data, sign=sign)

    perceptron = Perceptron(tol=1, alpha=0.9)
    perceptron.fit(x_train, y_train)
    predictions = perceptron.predict(x_test)
    return predictions, y_test


