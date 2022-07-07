import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import Config as c
from data import load_csv_data, load_custom_data, verify_data
from models import get_linreg_model, gen_algo, evaluate
from pre import preprocess_data, prepare_data, sort_splitted_data
from predictions import predict_ranged_days, prediction_date_based
from tables import compar_error, compar_error_plain, rekap_table
from plots import error_bar_chart, error_line_chart, predictions_line_chart
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
            size = st.number_input(label="Ukuran Populasi", min_value=10, step=10)
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
            population, fitness, linreg_ga = gen_algo(size=size, n_gen=n_gen, cr=cr, mr=mr, X_train=X_train, y_train=y_train, mode=mode)

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


    # Tampilan Hasil Evaluasi Model
    with st.expander("Hasil Evaluasi Model", expanded=True):
        if "linreg" in st.session_state:
            # Dapatkan mode
            mode = get_session("mode")
            
            # Dapatkan data train dan test
            X_train, X_test, y_train, y_test = get_session("X_train", "X_test", "y_train", "y_test")

            # Urutkan data
            X_train_sorted, X_test_sorted = sort_splitted_data(X_train, X_test)
            y_train_sorted, y_test_sorted = sort_splitted_data(y_train, y_test)

            # Dapatkan model regresi
            linreg, linreg_ga = get_session("linreg", "linreg_ga")

            # Dapatkan nilai fitness terbaik
            best_fitness = get_session("best_fitness")

            # Dapatkan scaler
            scaler_y = get_session("scaler_y")

            # Evaluasi model regresi linier
            r2, mse, rmse = evaluate(X_test_sorted, y_test_sorted, linreg, scaler_y)
            linreg_metrics = [r2, mse, rmse, None]

            # Evaluasi model regresi linier + GA
            r2_ga, mse_ga, rmse_ga = evaluate(X_test_sorted, y_test_sorted, linreg_ga, scaler_y)
            best_fitness = 1 / mse_ga  # comment code ini apabila ingin menggunakan data normal
            linreg_ga_metrics = [r2_ga, mse_ga, rmse_ga, best_fitness]

            st.write(f"Metrik regresi pada harga {mode}")

            # Tampilkan tabel metrik
            metric_table = pd.DataFrame(
                data=[linreg_metrics, linreg_ga_metrics],
                index=["Regresi Linier", "Regresi Linier + GA"],
                columns=["R2", "MSE", "RMSE", "Fitness"]
            )
            metric_table = metric_table.style.format(precision=7)
            st.table(metric_table)

            # Simpan metrik ke dalam session
            set_session(
                r2=r2, mse=mse, rmse=rmse,
                r2_ga=r2_ga, mse_ga=mse_ga, rmse_ga=rmse_ga,
                X_test_sorted=X_test_sorted, y_test_sorted=y_test_sorted
            )
                

    # Tampilan Hasil Perbandingan Prediksi
    with st.expander("Hasil Perbandingan Prediksi", expanded=True):
        if "linreg" in st.session_state:
            # Dapatkan mode
            mode = get_session("mode")

            # Dapatkan data train dan test
            X_train, X_test, y_train, y_test = get_session("X_train", "X_test", "y_train", "y_test")

            # Dapatkan data yang telah diurutkan
            X_test_sorted, y_test_sorted = get_session("X_test_sorted", "y_test_sorted")
            
            # Dapatkan scaler
            scaler_y = get_session("scaler_y")

            # Dapatkan model regresi
            linreg, linreg_ga = get_session("linreg", "linreg_ga")

            # Dapatkan metrik
            mse, rmse, mse_ga, rmse_ga = get_session("mse", "rmse", "mse_ga", "rmse_ga")

            st.write(f"Prediksi pada harga {mode}")

            # Dapatkan tabel rekapitulasi
            rekap = rekap_table(
                X_test=X_test_sorted,
                y_test=y_test_sorted,
                model=linreg, 
                model_ga=linreg_ga,
                scaler_y=scaler_y,
            )

            rekap_first_table = rekap[
                ["Y_test", "MLR Without Genetic", "MLR With Genetic", "Error MLR", "Error MLR+Genetic",]
            ]
            rekap_second_table = rekap[
                ["Y_test", "MLR Without Genetic", "MLR With Genetic", "Error MSE MLR", "Error MSE MLR+Genetic",]
            ]
            
            st.dataframe(rekap_first_table.style.format(precision=2))
            st.markdown("#")
            st.dataframe(rekap_second_table.style.format(precision=2))

            # Dapatkan rata-rata error
            # error_data = compar_error_plain(mse, mse_ga, rmse, rmse_ga)
            error_data = compar_error(rekap)
            
            for error_lable, value in error_data.items():
                st.write(f"{error_lable}: {value:.2f}")

            # Simpan rekap ke dalam session
            set_session(rekap=rekap)


    # Tampilan Visualisasi Error
    with st.expander("Visualisasi Error", expanded=True):
        if "linreg" in st.session_state:
            # Dapatkan mode
            mode = get_session("mode")

            # Dapatkan rekap
            rekap = get_session("rekap")

            # Tampilkan diagram batang error
            st.write(f"Diagram batang MSE pada harga {mode}")
            bar_chart = error_bar_chart(rekap)
            st.bokeh_chart(bar_chart)
            
            st.markdown("#")

            # Tampilkan diagram garis error
            st.write(f"Diagram garis MSE pada harga {mode}")
            bar_chart = error_line_chart(rekap)
            st.bokeh_chart(bar_chart)
            

    # Tampilan Prediksi Jangka Waktu Tertentu
    with st.expander("Prediksi Jangka Waktu Tertentu", expanded=True):
        if "linreg" in st.session_state:
            # Dapatkan mode
            mode = get_session("mode")

            # Dapatkan prediktor asli
            X_unshifted = get_session("X_unshifted")

            # Dapatkan scaler
            scaler_y = get_session("scaler_y")

            # Dapatkan model regresi
            linreg, linreg_ga = get_session("linreg", "linreg_ga")

            # Dapatkan rekap
            rekap = get_session("rekap")

            with st.form("Period"):
                period = st.number_input(label="Jangka Waktu Prediksi (hari)", min_value=5, max_value=30)
                is_submit = st.form_submit_button("Prediksi")
                st.markdown("#")

                if is_submit:
                    predict_period = predict_ranged_days(
                        period=period, 
                        X_unshifted=X_unshifted, 
                        rekap=rekap,
                        model=linreg,
                        model_ga=linreg_ga,
                        scaler_y=scaler_y
                    )
                    st.write(f"Prediksi harga {mode} pada jangka waktu {period} hari")
                    st.dataframe(predict_period.style.format(precision=0))
                    st.markdown("#")

                    # error_chart = error_bar_chart(predict_period, days=period)

                    value_chart = predictions_line_chart(predict_period)
                    st.write(f"Diagram garis harga {mode} pada jangka waktu {period} hari")
                    st.bokeh_chart(value_chart)
                    st.markdown("#")

                    if mode == c.BUY_MODE:
                        optimal_date = predict_period["MLR Without Genetic"].idxmin()
                        optimal_date_ga = predict_period["MLR With Genetic"].idxmin()
                    
                    elif mode == c.SELL_MODE:
                        optimal_date = predict_period["MLR Without Genetic"].idxmax()
                        optimal_date_ga = predict_period["MLR With Genetic"].idxmax()

                    st.markdown(f"- Jika menggunakan regresi biasa, disarankan untuk {mode} emas pada tanggal {optimal_date}")
                    st.markdown(f"- Jika menggunakan regresi + GA, disarankan untuk {mode} emas pada tanggal {optimal_date_ga}")





    # Tampilan Prediksi Tanggal Tertentu
    with st.expander("Prediksi Tanggal Tertentu", expanded=True):
        if "linreg" in st.session_state:
            # Dapatkan mode
            mode = get_session("mode")

            # Dapatkan prediktor asli
            X_unshifted = get_session("X_unshifted")

            # Dapatkan scaler
            scaler_y = get_session("scaler_y")

            # Dapatkan model regresi
            linreg, linreg_ga = get_session("linreg", "linreg_ga")

            # Tentukan tanggal minimal dan maksimal
            shift = c.SHIFT
            min_value=X_unshifted.index[shift]
            max_value=X_unshifted.index[-1] + pd.Timedelta(days=shift)

            with st.form("Date"):
                date = st.date_input(label="Masukkan Tanggal", value=min_value, min_value=min_value, max_value=max_value)
                is_submit = st.form_submit_button("Prediksi")

            st.markdown("#")
            
            if is_submit:
                predictions_date = prediction_date_based(
                    date=date, 
                    X=X_unshifted,
                    model=linreg,
                    model_ga=linreg_ga,
                    scaler_y=scaler_y,
                )
                st.write(f"Prediksi harga {mode} emas pada {date:%d %B %Y}")
                st.dataframe(predictions_date.style.format(precision=0))