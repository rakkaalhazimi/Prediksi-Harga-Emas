import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import Config as c
from data import load_csv_data, load_custom_data, verify_data
from models import get_linreg_model, gen_algo, evaluate, combine_predictions, prediction_date_based
from pre import preprocess_data, prepare_data
from visualization import compar_table, compar_error, error_bar_chart, error_line_chart, predictions_line_chart
from utils.sessions import get_session, set_session


def main():
    
    # Tampilan Home
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
    
    
    # Tampilan Tutorial
    st.subheader("Cara Penggunaan")
    st.markdown("""
    Prediksi Harga Emas dapat dilakukan dengan langkah-langkah sebagai berikut:
    - Memilih dataset asli atau custom dari pengguna. 
    - Memasukkan parameter training dan algoritma genetika.
    - Melatih model dengan parameter yang ditentukan.
    - Mulai Prediksi.

    #
    """)


    # Tampilan Tipe Dataset
    with st.expander("Tipe Dataset"):
        st.write("Pilih salah satu dari tipe dibawah")
        
        ori, custom = "Asli", "Custom"
        option = st.radio(label="Tipe Dataset", options=[ori, custom])
        
        base_data = load_csv_data(c.DATA_PATH)
        
        data_used = base_data
        dataset_type = ori

        if option == ori:
            st.markdown("Anda akan menggunakan dataset asli dari server")

        else:
            st.markdown("""
            Pastikan bahwa data memiliki:
            - 2 kolom dengan nama `HargaBeli` dan `HargaJual` yang berisikan bilangan cacah / bulat
            - 1 kolom dengan nama `Date` yang berisikan tanggal dengan format `DD/MM/YYYY`
            - Jumlah data lebih dari 20
            #
            """)
            user_file = st.file_uploader(label="Unggah file .csv, .xlsx", type=["csv", "xlsx"])
            custom_data = None
            valid = False

            if user_file:
                custom_data = load_custom_data(user_file)
                valid = verify_data(custom_data)

            if valid and custom_data is not None:
                data_used = custom_data
                dataset_type = custom

        st.write(dataset_type) 
            

    # Tampilan Parameter Data
    with st.expander("Parameter Data"):
        with st.form("Parameter data"):
            mode = st.selectbox(label="Pilihan Harga", options=[c.BUY_MODE, c.SELL_MODE])
            test_size = st.number_input(label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
            is_submit = st.form_submit_button("Simpan")
        
        if is_submit:
            # Siapkan data
            X, X_unshifted, y = prepare_data(data_used, mode=mode)

            # Proses data untuk machine learning
            X_train, X_test, X_unshifted, y_train, y_test, scaler_X, scaler_y = preprocess_data(X, X_unshifted, y, test_size)

            # Simpan data dalam session
            set_session(
                mode=mode,
                X_train=X_train, 
                X_test=X_test, 
                X_unshifted=X_unshifted, 
                y_train=y_train, 
                y_test=y_test,
                scaler_X=scaler_X,
                scaler_y=scaler_y
            )

    
    # Tampilan Parameter Genetika
    with st.expander("Parameter Genetika"):
        with st.form("Parameter gen"):
            n_gen = st.number_input(label="Jumlah Generasi", min_value=10, step=10)
            size = st.number_input(label="Ukuran Populasi", min_value=100, step=100)
            cr = st.number_input(label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
            mr = st.number_input(label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)
            is_submit = st.form_submit_button("Simpan")


    # Tampilan Latih Model
    with st.expander("Latih Model"):
        st.markdown("""
        Pastikan anda telah memilih tipe dataset dan parameter sebelum mulai melatih model.
        Jika sudah, tekan tombol `Latih` untuk mulai melatih model.
        """)

        # Tampilkan tabel pilihan user
        params_lable = [
            "Mode Harga",
            "Tipe Dataset", 
            "Ukuran Data Test", 
            "Jumlah Generasi", 
            "Ukuran Populasi", 
            "Crossover Rate", 
            "Mutation Rate"
        ]
        data = [mode, dataset_type, test_size, n_gen, size, cr, mr]
        data = list(map(str, data))
        param_table = pd.DataFrame(data, index=params_lable, columns=["Nilai"])
        st.table(param_table)

        is_train = st.button("Latih")
        st.markdown("#")

        if is_train and "X_train" in st.session_state:
            # Dapatkan data untuk regresi
            mode, X_train, y_train = get_session("mode", "X_train", "y_train")

            # Latih regresi linier
            linreg = get_linreg_model(X=X_train, y=y_train)

            # Latih regresi linier + GA
            population, fitness, linreg_ga = gen_algo(size=size, n_gen=n_gen, cr=cr, mr=mr, X_train=X_train, y_train=y_train)

            # Nilai fitness terbaik
            best_fitness = fitness[0]

            # Simpan hasil ke dalam session
            set_session(
                best_fitness=best_fitness,
                linreg=linreg,
                linreg_ga=linreg_ga
            )
            
        elif "linreg" in st.session_state and "X_train" in st.session_state:
            st.info("Model sudah dilatih")

        elif "X_train" in st.session_state:
            st.info("Model siap untuk dilatih")

        else:
            st.warning("Data belum disiapkan")

            

# def show_result_matric(mode):
#     col1, col2 = st.columns([6, 6])

#     with col1:
#         st.write("Metrik regresi linier pada harga {}".format(mode))
#         results = evaluate(st.session_state["linreg_{}".format(mode)], mode="{}".format(mode))
        
#         for metric in results:
#             st.write("{} : {:.3f}".format(metric.upper(), results[metric]))
#         st.write("--")
#         st.markdown("---")

#     with col2:
#         st.write("Metrik regresi linier + GA pada harga {}".format(mode))
#         results = evaluate(st.session_state["linreg_{}_ga".format(mode)], mode="{}".format(mode))
        
#         for metric in results:
#             st.write("{} : {:.3f}".format(metric.upper(), results[metric]))
#         st.write("Fitness {} : {:.4f}".format(mode, st.session_state["fitness_{}".format(mode)]))
#         st.markdown("---")


# @wrap_view("Hasil Evaluasi Model")
# @is_trained
# def view_result():
#     show_mode = st.selectbox("Jenis", ["Harga Jual", "Harga Beli", "Semua"], key="result")
#     show_mode = show_mode.lower().split()
#     st.markdown("#")
    
#     for mode in MODES:
#         if mode in show_mode or "semua" in show_mode:
#             show_result_matric(mode)
#         else:
#             continue



# def show_pred_comparison(mode):
#     st.markdown("**Prediksi pada harga {}**".format(mode))
#     rekap, rekap_show = compar_table(
#         model=st.session_state["linreg_{}".format(mode)], 
#         model_ga=st.session_state["linreg_{}_ga".format(mode)],
#         mode=mode,
#         **st.session_state["{}_real_test".format(mode)]
#     )
#     st.session_state["rekap_{}".format(mode)] = rekap
#     st.dataframe(rekap_show)

#     errors = compar_error(rekap)
#     for label in errors:
#         st.write("{} : {:.3f}".format(label, errors[label]))

#     st.markdown("#")


# @wrap_view(title="Perbandingan Prediksi")
# @is_trained
# def view_comparison():
#     show_mode = st.selectbox("Jenis", [ "Semua", "Harga Jual", "Harga Beli"], key="comparison")
#     st.markdown("#")
#     show_mode = show_mode.lower().split()
        
#     for mode in MODES:
#         if mode in show_mode or "semua" in show_mode:
#             show_pred_comparison(mode)
#         else:
#             continue
        

# def show_chart(mode):
#     chart_functs = [("Batang", error_bar_chart), ("Garis", error_line_chart)]
#     for shape, func in chart_functs:
#         st.markdown("**Diagram {} MSE pada harga {}**".format(shape, mode))
#         rekap = st.session_state["rekap_{}".format(mode)]
#         chart = func(rekap=rekap)
#         st.bokeh_chart(chart)
#     st.markdown("#")


# @wrap_view("Visualisasi Error")
# @is_trained
# def view_charts():
#     show_mode = st.selectbox("Jenis", [ "Semua", "Harga Jual", "Harga Beli"], key="charts")
#     st.markdown("#")
#     show_mode = show_mode.lower().split()
    
#     for mode in MODES:
#         if mode in show_mode or "semua" in show_mode:
#             show_chart(mode)
#         else:
#             continue
            


# def show_predict_period(mode, period):
#     predict_period = combine_predictions(
#         period=st.session_state["period"], 
#         X_test=st.session_state["predictor_{}".format(mode)], 
#         rekap=st.session_state["rekap_{}".format(mode)],
#         model=st.session_state["linreg_{}".format(mode)],
#         model_ga=st.session_state["linreg_{}_ga".format(mode)],
#         mode=mode
#     )
#     value_chart = predictions_line_chart(predict_period)
#     error_chart = error_bar_chart(predict_period, days=st.session_state["period"] * 2)

#     st.markdown("**Tabel prediksi harga {} pada jangka waktu {} hari**".format(mode, period))
#     st.dataframe(predict_period.style.format(precision=2))
#     st.markdown("##")
#     st.markdown("**Diagram garis harga {} pada jangka waktu {} hari**".format(mode, period))
#     st.bokeh_chart(value_chart)
#     st.markdown("**Diagram error {} pada jangka waktu {} hari**".format(mode, period))
#     st.bokeh_chart(error_chart)
#     st.markdown("#")


# @wrap_view("Prediksi Jangka Waktu Tertentu")
# @is_trained
# def view_predict_period():
#     with st.form("Period"):
#         show_mode = st.selectbox("Jenis", ["Semua", "Harga Jual", "Harga Beli"], key="result")
#         show_mode = show_mode.lower().split()
#         period = st.number_input(label="Jangka Waktu Prediksi (hari)", min_value=5, max_value=30)
#         is_submit = st.form_submit_button("Prediksi")
#     st.markdown("#")
    
#     if is_submit:
#         st.session_state["period"] = period
#     else:
#         return
    
#     for mode in MODES:
#         if mode in show_mode or "semua" in show_mode:
#             show_predict_period(mode, period)
#         else:
#             continue

        

# def show_predict_date(mode, date):
#     shift = st.session_state["shift"]
#     predictions_date = prediction_date_based(
#         date=date, 
#         X=st.session_state["predictor_{}".format(mode)],
#         model=st.session_state["linreg_beli"],
#         model_ga=st.session_state["linreg_beli_ga"],
#         mode=mode
#     )

#     st.write("Prediksi harga {} emas pada {:%d %B %Y}".format(mode, date))
#     st.dataframe(predictions_date.style.format(precision=2))


# @wrap_view("Prediksi Tanggal Tertentu")
# @is_trained
# def view_predict_date():
#     shift = st.session_state["shift"]
#     min_value=st.session_state["predictor_beli"].index[shift]
#     max_value=st.session_state["predictor_beli"].index[-1] + pd.Timedelta(days=shift)

#     with st.form("Date"):
#         show_mode = st.selectbox("Jenis", ["Semua", "Harga Jual", "Harga Beli"], key="result")
#         date = st.date_input(label="Masukkan Tanggal", value=min_value, min_value=min_value, max_value=max_value)
#         is_submit = st.form_submit_button("Prediksi")
#         show_mode = show_mode.lower().split()
    
#     if is_submit:
#         for mode in MODES:
#             if mode in show_mode or "semua" in show_mode:
#                 show_predict_date(mode, date)
#             else:
#                 continue