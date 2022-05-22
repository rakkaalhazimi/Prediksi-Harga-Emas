import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import Config as c


def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def init_scaler(data, scaler):
    scaler_instance = scaler()
    scaler_fitted = scaler_instance.fit(data)
    return scaler_fitted

def apply_test_scaler(X_test, y_test, mode):
    X_test = X_test.copy()
    y_test = y_test.copy()

    X_test[:] = st.session_state["scaler_{}_x".format(mode)].transform(X_test)
    y_test[:] = st.session_state["scaler_{}_y".format(mode)].transform(y_test)

    return {"X_test": X_test, "y_test": y_test}


def date_offset(df):
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d") + pd.Timedelta(days=st.session_state["shift"])
    df.index = [date.strftime("%Y-%m-%d") for date in df.index]


def preprocess_data(X, X_unshifted, y, test_size):
    # Split Data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    # Inisiasi Scaler
    scaler_X = init_scaler(X_train, MinMaxScaler)
    scaler_y = init_scaler(y_train, MinMaxScaler)

    # Terapkan Scaler
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_unshifted = scaler_X.transform(X_unshifted)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, X_unshifted, y_train, y_test, scaler_X, scaler_y



