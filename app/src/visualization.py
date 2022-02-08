import streamlit as st
import numpy as np
import pandas as pd


def compar_table(X_test, y_test, model, model_ga):
    y_test_series = np.squeeze(y_test.values)
    predictions_series = np.squeeze(model.predict(X_test))
    predictions_ga_series = np.squeeze(model_ga.predict(X_test))
    index = [date.strftime("%Y-%m-%d") for date in y_test.index]

    rekap = pd.DataFrame({
        "Y_test": y_test_series,
        "MLR Without Genetic": predictions_series,
        "MLR With Genetic": predictions_ga_series,
        "Error MLR": abs(y_test_series - predictions_series),
        "Error MLR+Genetic": abs(y_test_series - predictions_ga_series),
        "Error MSE MLR": (y_test_series - predictions_series)**2,
        "Error MSE MLR+Genetic": (y_test_series - predictions_ga_series)**2,
        "Error RMSE MLR": np.sqrt((y_test_series - predictions_series)**2),
        "Error RMSE MLR+Genetic": np.sqrt((y_test_series - predictions_ga_series)**2),
    }, index=index)

    # rekap = rekap.style.format(precision=2)
    return rekap