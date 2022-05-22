import streamlit as st

from config import Config as c
from utils.sessions import set_session
from utils.wrappers import wrap_view
from views.parameters.prepare import prepare_data
from views.parameters.mlprocess import preprocess_data


@wrap_view(title="Parameter Data")
def view_data_parameter():

    # Siapkan input yang diperlukan
    df = st.session_state["dataset"]

    with st.form("Parameter data"):
        st.write("Parameter data")
        mode = st.selectbox(label="Pilihan Harga", options=[c.BUY_MODE, c.SELL_MODE])
        test_size = st.number_input(label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
        is_submit = st.form_submit_button("Simpan")
    
    if is_submit:
        # Simpan parameter dalam session
        set_session(mode=mode, test_size=test_size)

        # Siapkan data
        X, X_unshifted, y = prepare_data(df, mode=mode)

        # Proses data untuk machine learning
        X_train, X_test, X_unshifted,\
        y_train, y_test,\
        scaler_X, scaler_y = preprocess_data(X, X_unshifted, y, test_size)

        # Simpan data dalam session
        set_session(
            X_train=X_train, 
            X_test=X_test, 
            X_unshifted=X_unshifted, 
            y_train=y_train, 
            y_test=y_test,
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )
        


@wrap_view(title="Parameter Genetika")
def view_gen_parameter():

    with st.form("Parameter gen"):
        st.write("Parameter Algoritma Genetika")
        n_gen = st.number_input(label="Jumlah Generasi", min_value=10, step=10)
        size = st.number_input(label="Ukuran Populasi", min_value=100, step=100)
        cr = st.number_input(label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        mr = st.number_input(label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)
        is_submit = st.form_submit_button("Simpan")

    if is_submit:
        # Simpan parameter dalam session
        set_session(n_gen=n_gen, size=size, cr=cr, mr=mr)