import streamlit as st

from config import Config as c
from utils.sessions import set_session
from utils.wrappers import wrap_view
from views.parameters.data_preparator import prepare_data


@wrap_view(title="Parameter")
def view_parameter():

    # Siapkan input yang diperlukan
    df = st.session_state["dataset"]

    with st.form("Parameter"):
        st.write("Parameter data")
        mode = st.selectbox(label="Pilihan Harga", options=[c.BUY_MODE, c.SELL_MODE])
        test_size = st.number_input(label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
        
        st.markdown("")

        st.write("Parameter Algoritma Genetika")
        n_gen = st.number_input(label="Jumlah Generasi", min_value=10, step=10)
        size = st.number_input(label="Ukuran Populasi", min_value=100, step=100)
        cr = st.number_input(label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        mr = st.number_input(label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)

        is_submit = st.form_submit_button("Simpan")
    
    if is_submit:
        # Simpan parameter dalam session
        set_session(mode=mode, test_size=test_size, n_gen=n_gen, size=size, cr=cr, mr=mr)

        # Siapkan data
        X, X_real, y = prepare_data(df, mode=mode)

        # beli_train, beli_test, jual_train, jual_test = preprocess_data(st.session_state["dataset"], st.session_state["test"])
        
        # st.session_state["beli_train"] = beli_train
        # st.session_state["beli_test"] = beli_test
        # st.session_state["jual_train"] = jual_train
        # st.session_state["jual_test"] = jual_test

        # len_test = len(st.session_state["beli_test"]["X_test"])
        # st.session_state["beli_real_test"] = {
        #     "X_test": st.session_state["X_beli"].iloc[-len_test:], 
        #     "y_test": st.session_state["y_beli"].iloc[-len_test:]
        # }
        # st.session_state["jual_real_test"] = {
        #     "X_test": st.session_state["X_jual"].iloc[-len_test:], 
        #     "y_test": st.session_state["y_jual"].iloc[-len_test:]
        # }

        # st.session_state["beli_real_test"] = apply_test_scaler(**st.session_state["beli_real_test"], mode="beli")
        # st.session_state["jual_real_test"] = apply_test_scaler(**st.session_state["jual_real_test"], mode="jual")