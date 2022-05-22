from sklearn.linear_model import LinearRegression


def get_linreg_model(X, y):
    linreg = LinearRegression()
    linreg = linreg.fit(X, y)
    return linreg