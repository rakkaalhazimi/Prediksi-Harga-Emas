import pandas as pd
import streamlit as st

from config import Config as c


def convert_to_datetime(df, colname, format):
    df[colname] = pd.to_datetime(df[colname], format=format)
    return df

def reverse_df(df):
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    return df

def scale_data(df):
    return df / 1000

def get_variables(df, mode, period=1):
    # Copy dataframe supaya tidak menimpa dataframe asli
    df = df.copy()

    # Ubah huruf pertama mode menjadi kapital
    mode = mode.title()

    # Tentukan variabel
    predictor = df[["Inflasi", "HargaMinyak", f"Kurs{mode}"]]
    respon = df[[f"Harga{mode}"]]

    # Mundurkan variabel respon ke observasi pada beberapa hari yang lalu
    respon = respon.shift(-period)

    # Hilangkan missing value
    respon = respon.dropna()

    # Simpan predictor asli
    st.session_state["predictor_{}".format(mode.lower())] = predictor
    predictor_unshifted = predictor.copy()

    # Sejajarkan variabel respon dan prediktor supaya memiliki jumlah observasi yang sama
    predictor = predictor.loc[respon.index]

    # Sesuaikan tanggal pada variabel respon
    respon.index += pd.Timedelta(days=period)

    return predictor, predictor_unshifted, respon


def prepare_data(df, mode):
    # Pre - Urutkan Data
    df = reverse_df(df)

    # Pre - Mengubah indeks menjadi tanggal
    df = convert_to_datetime(df=df, colname=c.DATE_COL, format=c.DATE_FORMAT)

    # Pre - Gunakan kolom tanggal sebagai indeks
    df = df.set_index(c.DATE_COL)

    # Pre - Ambil variabel dari Data
    X, X_real, y = get_variables(df, mode, period=c.SHIFT)

    # Pre - Atur Skala Data
    y = scale_data(y)

    return X, X_real, y