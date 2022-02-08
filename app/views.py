from abc import abstractmethod
from typing import Any
import streamlit as st
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.visualization import compar_table, error_bar_chart, error_line_chart, predictions_line_chart



class ViewElement:

    @abstractmethod
    def build(self) -> Any:
        ...


class Component:
    def __init__(self, comp, *args, **kwargs) -> None:
        self.comp = comp
        self.args = args
        self.kwargs = kwargs

    def show(self) -> None:
        return self.comp(*self.args, **self.kwargs)


class Header(ViewElement):
    def __init__(self, title, desc) -> None:
        self.title = Component(st.title, body=title)
        self.desc = Component(st.write, desc)
        self.comps = [self.title, self.desc]

    def build(self) -> None:
        for comp in self.comps:
            comp.show()


class InfoBoard(ViewElement):
    def __init__(self, title, desc) -> None:
        self.title = Component(st.subheader, body=title)
        self.desc = Component(st.write, desc)
        self.comps = [self.title, self.desc]

    def build(self) -> None:
        st.markdown("#")
        for comp in self.comps:
            comp.show()
        st.markdown("---")


class InfoBoardWithButton(ViewElement):
    def __init__(self, title, desc, btn_label) -> None:
        self.title = Component(st.subheader, body=title)
        self.desc = Component(st.write, desc)
        self.button = Component(st.button, btn_label)
        self.comps = [self.title, self.desc]

    def build(self) -> bool:
        st.markdown("#")
        for comp in self.comps:
            comp.show()

        is_train = self.button.show()
        
        return is_train


class PreParam(ViewElement):
    def __init__(self) -> None:
        self.title = Component(st.subheader, body="Setelan Jumlah Data Test")
        self.desc = Component(st.write, 
        """Tentukan proporsi antara jumlah data latih dan data test.
        """
        )
        self.test_size = Component(st.number_input, label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
        self.submit = Component(st.form_submit_button, label="Konfirmasi")

        self.comps = [self.test_size]
        self.pnames = ["test_size"]

    def build(self) -> dict:
        st.markdown("#")
        self.title.show()
        self.desc.show()

        params = {}
        with st.form("ParameterTest"):
            for key, comp in zip(self.pnames, self.comps):
                val = comp.show()
                params[key] = val
        
            is_submit = self.submit.show()

        if is_submit:
            return params


class GAParam(ViewElement):
    def __init__(self) -> None:
        self.title = Component(st.subheader, body="Setelan Algoritma Genetika")
        self.desc = Component(st.write, 
        """Tentukan nilai parameter untuk melatih model 
        menggunakan metode regresi linier dengan optimalisasi algoritma genetika.
        Setelah parameter ditentukan, klik 'Konfirmasi'.
        """
        )
        self.generation = Component(st.number_input, label="Jumlah Generasi", min_value=10, max_value=100, step=10)
        self.size = Component(st.number_input, label="Ukuran Populasi", min_value=100, max_value=1500, step=100)
        self.cr = Component(st.number_input, label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        self.mr = Component(st.number_input, label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)
        self.submit = Component(st.form_submit_button, label="Konfirmasi")

        self.comps = [self.generation, self.size, self.cr, self.mr]
        self.pnames = ["n_gen", "size", "cr", "mr"]

    def build(self) -> dict:
        st.markdown("#")
        self.title.show()
        self.desc.show()

        params = {}
        with st.form("ParameterGA"):
            for key, comp in zip(self.pnames, self.comps):
                val = comp.show()
                params[key] = val
        
            is_submit = self.submit.show()

        if is_submit:
            return params


class MetricsReport(ViewElement):
    def __init__(self, title, model, X_test, y_test) -> None:
        self.title = title
        predictions = model.predict(X_test)
        r2 = r2_score(predictions, y_test)
        mse = mean_squared_error(predictions, y_test)
        rmse = mean_squared_error(predictions, y_test, squared=False)
        coef = model.coef_
        intercept = model.intercept_

        self.r2 = Component(st.metric, label="R2 Score", value="{:.2%}".format(r2))
        self.mse = Component(st.metric, label="MSE Score", value="{:.2f}".format(mse))
        self.rmse = Component(st.metric, label="RMSE Score", value="{:.2f}".format(rmse))
        self.coef = Component(st.write, "Koefisien")
        self.intercept = Component(st.write, "Intersep")
         
        self.comps = [self.r2, self.mse, self.rmse]


    def build(self) -> None:
        with st.expander(self.title):
            for comp in self.comps:
                comp.show()


class ComparationReport(ViewElement):
    def __init__(self, title, X_test, y_test, model, model_ga) -> None:
        self.title = title
        self.rekap = compar_table(X_test, y_test, model, model_ga)
        mean_mse_error = self.rekap["Error MSE MLR"].mean()
        mean_rmse_error = np.sqrt(mean_mse_error)
        mean_ga_mse_error = self.rekap["Error MSE MLR+Genetic"].mean()
        mean_ga_rmse_error = np.sqrt(mean_ga_mse_error)

        self.table = Component(st.dataframe, self.rekap.style.format(precision=2))
        self.mse = Component(st.metric, label="Rata-rata error MSE tanpa algoritma genetika", value="{:.2f}".format(mean_mse_error))
        self.mse_ga = Component(st.metric, label="Rata-rata error MSE dengan algoritma genetika", value="{:.2f}".format(mean_ga_mse_error))
        self.rmse = Component(st.metric, label="Rata-rata error RMSE tanpa algoritma genetika", value="{:.2f}".format(mean_rmse_error))
        self.rmse_ga = Component(st.metric, label="Rata-rata error RMSE dengan algoritma genetika", value="{:.2f}".format(mean_ga_rmse_error))
        self.comps = [self.table, self.mse, self.mse_ga, self.rmse, self.rmse_ga]


    def build(self) -> Any:
        with st.expander(self.title):
            for comp in self.comps:
                comp.show()

        return self.rekap


class BarChartError(ViewElement):
    def __init__(self, title, rekap) -> None:
        self.title = title
        self.rekap = rekap

    def build(self) -> None:
        chart = error_bar_chart(rekap=self.rekap)
        with st.expander(self.title):
            st.bokeh_chart(chart)


class LineChartError(ViewElement):
    def __init__(self, title, rekap) -> None:
        self.title = title
        self.rekap = rekap

    def build(self) -> None:
        chart = error_line_chart(rekap=self.rekap)
        with st.expander(self.title):
            st.bokeh_chart(chart)


class PredictionBoard(ViewElement):
    def __init__(self, title, desc) -> None:
        self.title = Component(st.subheader, title)
        self.desc = Component(st.write, desc)
        self.period = Component(st.number_input, label="Jangka Waktu Prediksi (hari)", min_value=5, max_value=30)
        self.submit = Component(st.form_submit_button, label="Prediksi")

        self.comps = [self.period]
        self.pnames = ["period"]

    def build(self) -> dict:
        st.markdown("#")
        self.title.show()
        self.desc.show()

        params = {}

        with st.form("ParameterPeriod"):
            for key, comp in zip(self.pnames, self.comps):
                val = comp.show()
                params[key] = val

            is_submit = self.submit.show()

        if is_submit:
            return params


class PredictionReport(ViewElement):
    def __init__(self, title, predictions) -> None:
        self.title = title
        chart = predictions_line_chart(predictions)
        self.predictions = Component(st.dataframe, predictions.style.format(precision=2))
        self.chart = Component(st.bokeh_chart, chart)
        self.comps = [self.chart, self.predictions]


    def build(self) -> Any:
        with st.expander(self.title):
            for comp in self.comps:
                comp.show()