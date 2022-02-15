import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from views import ( 
    view_home, view_tutorial, view_dataset_type, view_parameter, view_train,
    view_result, view_comparison
    )
from styles import css_style
from src.data import load_data
from src.models import gen_algo, combine_predictions, prediction_date_based
from src.pre import preprocess_data


css_style()



view_home()
view_tutorial()
view_dataset_type()
view_parameter()
view_train()
view_result()
view_comparison()

# View - Bagian Pembuka
# view_home()

# # View - Info Dataset
# model_info = InfoBoard(
#     title="Dataset", 
#     desc="Data yang digunakan adalah data harga beli emas dan harga jual emas pada tanggal xx sampai xx")
# model_info.build()

# # View - Info Model
# model_info = InfoBoard(
#     title="Informasi Model", 
#     desc="Model yang digunakan adalah regresi linier dan regresi linier dengan algoritma genetika")
# model_info.build()

# # View - Parameter Praproses Data
# test_param = PreParam()
# test_size_input = test_param.build()


# # Pre - Praproses Data
# if test_size_input:
#     beli_train, beli_test, jual_train, jual_test = preprocess_data(df, test_size_input["test_size"])
#     st.session_state["test_size"] = test_size_input["test_size"]

# elif st.session_state.get("test_size"):
#     beli_train, beli_test, jual_train, jual_test = preprocess_data(df, st.session_state["test_size"])


# # View - Parameter Algoritma Genetika
# ga_param = GAParam()
# ga_input = ga_param.build()

# if ga_input:
#     st.session_state.update(ga_input)


# # View - Latih Model
# train_info = InfoBoardWithButton(
#     title="Latih Model",
#     desc="Lakukan pelatihan pada model regresi linier biasa dan regresi linier dengan optimalisasi algoritma genetika",
#     btn_label="Latih"
#     )
# is_train = train_info.build()


# has_set_params = st.session_state.get("size") and st.session_state.get("test_size") and is_train

# if has_set_params:
#     # Latih Regresi Linier
#     linreg_beli = LinearRegression().fit(beli_train["X_train"], beli_train["y_train"])
#     linreg_jual = LinearRegression().fit(jual_train["X_train"], jual_train["y_train"])

#     # Latih Regresi Linier dengan GA
#     st.markdown("#")
#     ga_input = {
#         "size": st.session_state.get("size"),
#         "n_gen": st.session_state.get("n_gen"),
#         "cr": st.session_state.get("cr"),
#         "mr": st.session_state.get("mr"),
#     }
#     population_beli, fitness_beli, linreg_beli_ga = gen_algo(**ga_input, **beli_train)
#     population_jual, fitness_jual, linreg_jual_ga = gen_algo(**ga_input, **jual_train)

#     # Simpan hasil train ke dalam session
#     st.session_state["populasi_beli"] = population_beli
#     st.session_state["populasi_jual"] = population_jual
#     st.session_state["fitness_beli"]  = fitness_beli
#     st.session_state["fitness_jual"]  = fitness_jual
#     st.session_state["linreg_beli_ga"]  = linreg_beli_ga
#     st.session_state["linreg_jual_ga"]  = linreg_jual_ga
#     st.session_state["linreg_beli"]  = linreg_beli
#     st.session_state["linreg_jual"]  = linreg_jual


# # st.session_state

#     # View - Laporan Hasil Latihan
#     st.subheader("Evaluasi Model")
#     error_linreg_beli = MetricsReport(
#         title="Hasil Metric Regresi Linier pada Harga Beli",
#         model=st.session_state["linreg_beli"],
#         **beli_test
#         )
#     error_linreg_beli.build()

#     error_linreg_jual = MetricsReport(
#         title="Hasil Metric Regresi Linier pada Harga Jual",
#         model=st.session_state["linreg_jual"],
#         **jual_test
#         )
#     error_linreg_jual.build()

#     error_linreg_ga_beli = MetricsReport(
#         title="Hasil Metric Regresi Linier + GA pada Harga Beli",
#         model=st.session_state["linreg_beli_ga"],
#         **beli_test
#         )
#     error_linreg_ga_beli.build()

#     error_linreg_ga_jual = MetricsReport(
#         title="Hasil Metric Regresi Linier + GA pada Harga Jual",
#         model=st.session_state["linreg_jual_ga"],
#         **jual_test
#         )
#     error_linreg_ga_jual.build()
#     st.markdown("#")

#     # View - Laporan Hasil Perbandingan
#     st.subheader("Nilai Fitness")
#     fitness_beli_report = FitnessReport(
#         title="Nilai Fitness pada Harga Beli", 
#         label="Fitness", 
#         fitness=st.session_state["fitness_beli"][0]
#         )
#     fitness_beli_report.build()

#     fitness_jual_report = FitnessReport(
#         title="Nilai Fitness pada Harga Jual", 
#         label="Fitness", 
#         fitness=st.session_state["fitness_jual"][0]
#         )
#     fitness_jual_report.build()
#     st.markdown("#")

#     # View - Laporan Hasil Perbandingan
#     st.subheader("Perbandingan")
#     compar_beli = ComparationReport(
#         title="Perbandingan pada Harga Beli",
#         model=st.session_state["linreg_beli"], 
#         model_ga=st.session_state["linreg_beli_ga"],
#         **beli_test)
#     rekap_beli = compar_beli.build()
#     st.session_state["rekap_beli"] = rekap_beli

#     compar_jual = ComparationReport(
#         title="Perbandingan pada Harga Jual",
#         model=st.session_state["linreg_jual"], 
#         model_ga=st.session_state["linreg_jual_ga"],
#         **jual_test)
#     rekap_jual = compar_jual.build()
#     st.session_state["rekap_jual"] = rekap_jual
#     st.markdown("#")


#     # View - Visualisasi Diagram Garis dan Batang Error
#     st.subheader("Visualisasi Error")
#     bar_chart_beli = BarChartError(title="Diagram Batang Error pada Harga Beli", rekap=rekap_beli)
#     bar_chart_beli.build()

#     bar_chart_jual = BarChartError(title="Diagram Batang Error pada Harga Jual", rekap=rekap_jual)
#     bar_chart_jual.build()

#     line_chart_beli = LineChartError(title="Diagram Garis Error pada Harga Beli", rekap=rekap_beli)
#     line_chart_beli.build()

#     line_chart_jual = LineChartError(title="Diagram Garis Error pada Harga Jual", rekap=rekap_jual)
#     line_chart_jual.build()


# # View - Prediksi Jangka Waktu Tertentu
# papan_prediksi = PredictionBoard(
#     title="Prediksi Masa Depan pada Jangka Waktu Tertentu", 
#     desc="Masukkan berapa jangka waktu untuk diprediksi")
# periode_input = papan_prediksi.build()

# if periode_input and st.session_state.get("rekap_beli") is not None:
#     st.session_state.update(periode_input)
    
#     predict_period_beli = combine_predictions(
#         period=st.session_state["period"], 
#         X_test=beli_test["X_test"], 
#         rekap=st.session_state["rekap_beli"],
#         model=st.session_state["linreg_beli"],
#         model_ga=st.session_state["linreg_beli_ga"])

#     predict_period_jual = combine_predictions(
#         period=st.session_state["period"], 
#         X_test=beli_test["X_test"], 
#         rekap=st.session_state["rekap_jual"],
#         model=st.session_state["linreg_jual"],
#         model_ga=st.session_state["linreg_jual_ga"])

#     predict_beli_depan = PredictionReport(
#         title="Prediksi Harga Beli pada Jangka Waktu {} hari".format(st.session_state["period"]),
#         predictions=predict_period_beli)
#     predict_beli_depan.build()

#     predict_beli_depan = PredictionReport(
#         title="Prediksi Harga Jual pada Jangka Waktu {} hari".format(st.session_state["period"]),
#         predictions=predict_period_jual)
#     predict_beli_depan.build()

# # View - Prediksi pada Tanggal Tertentu
# if st.session_state.get("rekap_beli") is not None:
#     date_predicts = DatePredictionBoard(
#         title="Prediksi pada Tanggal Tertentu", 
#         desc="Masukkan tanggal prediksi",
#         min_value=beli_train["y_train"].index[0],
#         max_value=beli_test["y_test"].index[-1] + pd.Timedelta(days=30)
#         )

#     date_input = date_predicts.build()

#     if date_input:
#         predictions_date = prediction_date_based(
#             date=date_input["date"], 
#             X=beli_train["X_train"].append(beli_test["X_test"]),
#             y=beli_train["y_train"].append(beli_test["y_test"]),
#             model=st.session_state["linreg_beli"],
#             model_ga=st.session_state["linreg_beli_ga"]
#             )

#         st.write("Prediksi pada {:%d %B %Y}".format(date_input["date"]))
#         predictions_date