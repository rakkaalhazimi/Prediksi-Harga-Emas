import streamlit as st

from config import Config as c
from utils.sessions import get_session, set_session
from utils.wrappers import wrap_view
from views.train.ga import gen_algo
from views.train.linear import get_linreg_model
from views.train.table import get_param_table


@wrap_view("Latih Model")
def view_train():
    st.markdown("""
    Pastikan anda telah memilih tipe dataset dan parameter sebelum mulai melatih model.
    Jika sudah, tekan tombol `Latih` untuk mulai melatih model.
    """)

    # Tampilkan tabel pilihan user
    param_table = get_param_table(params_key=c.PARAMS_KEY, params_lable=c.PARAMS_LABLE)
    st.table(param_table)

    is_train = st.button("Latih")

    st.markdown("#")

    if is_train and not st.session_state.get("test_size"):
        st.warning("Parameter belum ditentukan")

    elif is_train and st.session_state.get("test_size"):
        
        # Dapatkan data untuk regresi
        X_train, X_test, y_train, y_test = get_session("X_train", "X_test", "y_train", "y_test")

        # Latih regresi linier
        linreg = get_linreg_model(X=X_train, y=y_train)

        # Dapatkan parameter GA
        size, n_gen, cr, mr = get_session("size", "n_gen", "cr", "mr")

        # Latih regresi linier + GA
        population, fitness, linreg_ga = gen_algo(
            size=size, 
            n_gen=n_gen, 
            cr=cr, 
            mr=mr, 
            X_train=X_train, 
            y_train=y_train
        )

        # Nilai fitness terbaik
        best_fitness = fitness[0]

        # Simpan hasil ke dalam session
        set_session(
            best_fitness=best_fitness,
            linreg=linreg,
            linreg_ga=linreg_ga
        )