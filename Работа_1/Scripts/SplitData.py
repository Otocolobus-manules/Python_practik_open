from sklearn.model_selection import train_test_split


def split_data(data, sign) -> tuple:
    y = data[sign]
    x = data.drop(sign, axis=1)

    x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=0.2, random_state=1000)
    return x_tr, y_tr, x_test, y_test
