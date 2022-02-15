from abc import abstractmethod
from typing import Any

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import load_data
from src.models import gen_algo, evaluate
from src.pre import preprocess_data
from src.visualization import compar_table, error_bar_chart, error_line_chart, predictions_line_chart


# Inisiasi
df = load_data()
session = st.session_state


def wrap_view(title):
    def decorate(func):
        def content_view():
            with st.expander(title):
                func()
        return content_view
    return decorate
    

def view_home():
    st.title("Prediksi Harga Emas")
    st.markdown("""
    ---
    Aplikasi untuk melakukan prediksi pada harga beli dan harga jual emas. Model machine learning
    yang digunakan adalah regresi linier dan regresi linier dengan optimalisasi algoritma genetika.
    Pelatihan model dilakukan dengan menggunakan dataset harga emas pada kurun waktu `1 Januari 2017`
    hingga `31 Juli 2021`.

    Adapun fitur-fitur yang terdapat pada aplikasi ini adalah:
    - Prediksi harga emas pada jangka waktu tertentu.
    - Prediksi harga emas pada tanggal tertentu.

    #
    """)


def view_tutorial():
    st.subheader("Cara Penggunaan")
    st.markdown("""
    Prediksi Harga Emas dapat dilakukan dengan langkah-langkah sebagai berikut:
    - Memilih dataset asli atau custom dari pengguna. 
    - Memasukkan parameter training dan algoritma genetika.
    - Melatih model dengan parameter yang ditentukan.
    - Mulai Prediksi.

    #
    """)


@wrap_view(title="Tipe Dataset")
def view_dataset_type():
    st.write("Pilih salah satu dari tipe dibawah")
    option = st.radio(label="Tipe Dataset", options=["Asli", "Custom"])

    if option == "Custom":
        st.markdown("""
        Pastikan bahwa data memiliki:
        - 2 kolom dengan nama `HargaBeli` dan `HargaJual` yang berisikan bilangan cacah / bulat
        - 1 kolom dengan nama `Date` yang berisikan tanggal dengan format `YYYY-MM-DD`
        - Jumlah data lebih dari 10
        #
        """)
        csv_file = st.file_uploader(label="Unggah file `.csv`", type="csv")
        if csv_file:
            session["custom_dataset"] = csv_file
            session["dataset_type"] = "Custom"
    
    else:
        st.write("Anda akan menggunakan dataset asli dari server")
        session["dataset_type"] = "Asli"
        session["custom_dataset"] = None


@wrap_view(title="Parameter")
def view_parameter():
    with st.form("Parameter"):
        st.write("Parameter data")
        test_size = st.number_input(label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
        st.markdown("")

        st.write("Parameter Algoritma Genetika")
        generation = st.number_input(label="Jumlah Generasi", min_value=10, step=10)
        size = st.number_input(label="Ukuran Populasi", min_value=100, step=100)
        cr = st.number_input(label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        mr = st.number_input(label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)


        is_submit = st.form_submit_button("Simpan")
    
    if is_submit:
        session["test"] = test_size
        session["n_gen"] = generation
        session["size"] = size
        session["cr"] = cr
        session["mr"] = mr

        beli_train, beli_test, jual_train, jual_test = preprocess_data(df, session["test"])
        
        session["beli_train"] = beli_train
        session["beli_test"] = beli_test
        session["jual_train"] = jual_train
        session["jual_test"] = jual_test


@wrap_view("Latih Model")
def view_train():
    prerequisites = ["dataset_type", "test", "n_gen", "size", "cr", "mr"]
    data = [str(session.get(req, "-belum ditentukan-")) for req in prerequisites]
    index = ["Tipe Dataset", "Ukuran Data Test", "Jumlah Generasi", "Ukuran Populasi", "Crossover Rate", "Mutation Rate"]
    
    st.markdown("""
    Pastikan anda telah memilih tipe dataset dan parameter sebelum mulai melatih model.
    Jika sudah, tekan tombol `Latih` untuk mulai melatih model.
    """)
    st.table(pd.DataFrame(data, index=index, columns=["Nilai"]))

    is_train = st.button("Latih")
    st.markdown("#")

    if is_train and not session.get("test"):
        st.warning("Parameter belum ditentukan")

    elif is_train and session.get("test"):
        # Train Linreg
        linreg_beli = LinearRegression().fit(session["beli_train"]["X_train"], session["beli_train"]["y_train"])
        linreg_jual = LinearRegression().fit(session["jual_train"]["X_train"], session["jual_train"]["y_train"])

        # Train GA
        ga_input = dict(size=session["size"], 
                        n_gen=session["n_gen"], 
                        cr=session["cr"], 
                        mr=session["mr"])

        population_beli, fitness_beli, linreg_beli_ga = gen_algo(**ga_input, **session["beli_train"], mode="beli")
        population_jual, fitness_jual, linreg_jual_ga = gen_algo(**ga_input, **session["jual_train"], mode="jual")

        session["fitness_beli"] = fitness_beli[0]
        session["fitness_jual"] = fitness_jual[0]
        session["linreg_beli"] = linreg_beli
        session["linreg_jual"] = linreg_jual
        session["linreg_beli_ga"] = linreg_beli_ga
        session["linreg_jual_ga"] = linreg_jual_ga





@wrap_view("Hasil Evaluasi Model")
def view_result():

    if session.get("linreg_beli"):
        col1, col2 = st.columns([6, 6])

        linreg = [session.get("linreg_beli"), session.get("linreg_jual")]
        linreg_ga = [session.get("linreg_beli_ga"), session.get("linreg_jual_ga")]
        modes = ["beli_test", "jual_test"]

        for model, model_ga, mode in zip(linreg, linreg_ga, modes):
            with col1:
                st.write("Metrik regresi linier pada harga {}".format(mode[:4]))
                results = evaluate(model, mode=mode)
                for metric in results:
                    st.write("{} : {:.3f}".format(metric.upper(), results[metric]))
                st.markdown("---")

            with col2:
                st.write("Metrik regresi linier + GA pada harga {}".format(mode[:4]))
                results = evaluate(model_ga, mode=mode)
                for metric in results:
                    st.write("{} : {:.3f}".format(metric.upper(), results[metric]))
                st.markdown("---")
            


    # warnings = {
    #     "dataset_type": "Tipe Dataset",
    #     "test": "Parameter"
    # }
    # prerequisites = ["dataset_type", "test"]
    # for req in prerequisites:
    #     if not session.get(req):
    #         st.warning("{} belum ditentukan.".format(warnings[req]))
    #         break

    # else:
    #     st.subheader("Latih Model")

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
        self.generation = Component(st.number_input, label="Jumlah Generasi", min_value=10, step=10)
        self.size = Component(st.number_input, label="Ukuran Populasi", min_value=100, step=100)
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


class FitnessReport(ViewElement):
    def __init__(self, title, label, fitness) -> None:
        self.title = title
        self.fitness = Component(st.metric, label=label, value=fitness)
        self.comps = [self.fitness]

    def build(self) -> None:
        with st.expander(self.title):
            for comp in self.comps:
                comp.show()


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


class DatePredictionBoard(ViewElement):
    def __init__(self, title, desc, min_value, max_value) -> None:
        self.title = Component(st.subheader, title)
        self.desc = Component(st.write, desc)
        self.date = Component(st.date_input, label="Masukkan Tanggal", value=min_value, min_value=min_value, max_value=max_value)
        self.submit = Component(st.form_submit_button, label="Prediksi")

        self.comps = [self.date]
        self.pnames = ["date"]

    def build(self) -> dict:
        st.markdown("#")
        self.title.show()
        self.desc.show()

        params = {}

        with st.form("ParameterDate"):
            for key, comp in zip(self.pnames, self.comps):
                val = comp.show()
                params[key] = val

            is_submit = self.submit.show()

        if is_submit:
            return params