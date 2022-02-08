import streamlit as st
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import Legend
from bokeh.palettes import Category10_4
from bokeh.transform import dodge


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


def error_bar_chart(rekap, days=30):
    different = rekap[["Error MSE MLR", "Error MSE MLR+Genetic"]].iloc[:days]
    dates_str = list(different.index)
    different["date"] = dates_str

    p = figure(width=900, height=500, x_range=dates_str, sizing_mode="stretch_width")

    v1 = p.vbar(x=dodge("date", -0.11, range=p.x_range), width=0.2, top="Error MSE MLR", 
                color=Category10_4[0], source=different)

    v2 = p.vbar(x=dodge("date", 0.11, range=p.x_range), width=0.2, top="Error MSE MLR+Genetic", 
                color=Category10_4[1], source=different)

    legend = Legend(items=[
        ("Error MSE MLR", [v1]), 
        ("Error MSE MLR+Genetic", [v2])
    ], location="left")

    p.xaxis.major_label_orientation = "vertical"
    p.add_layout(legend, "above")

    return p