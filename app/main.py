import streamlit as st
from sklearn.linear_model import LinearRegression
from views import InfoBoard, InfoBoardWithButton, GAParam, PreParam, Header, MetricsReport
from src.data import load_data
from src.models import gen_algo
from src.pre import preprocess_data

# Data - Muat Data
df = load_data()


# View - Bagian Pembuka
header = Header(
    title="Prediksi Harga Emas", 
    desc="Aplikasi untuk melakukan prediksi pada harga beli atau harga jual emas"
)
header.build()
st.markdown("---")

# View - Info Dataset
model_info = InfoBoard(
    title="Dataset", 
    desc="Data yang digunakan adalah data harga beli emas dan harga jual emas pada tanggal xx sampai xx")
model_info.build()

# View - Info Model
model_info = InfoBoard(
    title="Informasi Model", 
    desc="Model yang digunakan adalah regresi linier dan regresi linier dengan algoritma genetika")
model_info.build()

# View - Parameter Praproses Data
test_param = PreParam()
test_size_input = test_param.build()


# Pre - Praproses Data
if test_size_input:
    beli_train, beli_test, jual_train, jual_test = preprocess_data(df, test_size_input["test_size"])
    st.session_state["test_size"] = test_size_input["test_size"]

elif st.session_state.get("test_size"):
    beli_train, beli_test, jual_train, jual_test = preprocess_data(df, st.session_state["test_size"])


# View - Parameter Algoritma Genetika
ga_param = GAParam()
ga_input = ga_param.build()

if ga_input:
    st.session_state.update(ga_input)


# View - Latih Model
train_info = InfoBoardWithButton(
    title="Latih Model",
    desc="Lakukan pelatihan pada model regresi linier biasa dan regresi linier dengan optimalisasi algoritma genetika",
    btn_label="Latih"
    )
is_train = train_info.build()

if st.session_state.get("size") and st.session_state.get("test_size") and is_train:
    # Latih Regresi Linier
    linreg_beli = LinearRegression().fit(beli_train["X_train"], beli_train["y_train"])
    linreg_jual = LinearRegression().fit(jual_train["X_train"], jual_train["y_train"])

    # Latih Regresi Linier dengan GA
    st.markdown("#")
    ga_input = {
        "size": st.session_state.get("size"),
        "n_gen": st.session_state.get("n_gen"),
        "cr": st.session_state.get("cr"),
        "mr": st.session_state.get("mr"),
    }
    population_beli, fitness_beli, linreg_beli_ga = gen_algo(**ga_input, **beli_train)
    population_jual, fitness_jual, linreg_jual_ga = gen_algo(**ga_input, **beli_train)

    # Simpan hasil train ke dalam session
    st.session_state["populasi_beli"] = population_beli
    st.session_state["populasi_jual"] = population_jual
    st.session_state["fitness_beli"]  = fitness_beli
    st.session_state["fitness_jual"]  = fitness_jual
    st.session_state["linreg_beli_ga"]  = linreg_beli_ga
    st.session_state["linreg_jual_ga"]  = linreg_jual_ga
    st.session_state["linreg_beli"]  = linreg_beli
    st.session_state["linreg_jual"]  = linreg_jual


# st.session_state

    # View - Laporan Hasil Latihan
    st.subheader("Evaluasi Model")
    error_linreg_beli = MetricsReport(
        title="Hasil Metric Regresi Linier pada Harga Beli",
        model=st.session_state["linreg_beli"],
        **beli_test
        )
    error_linreg_beli.build()

    error_linreg_jual = MetricsReport(
        title="Hasil Metric Regresi Linier pada Harga Jual",
        model=st.session_state["linreg_jual"],
        **jual_test
        )
    error_linreg_jual.build()

    error_linreg_ga_beli = MetricsReport(
        title="Hasil Metric Regresi Linier + GA pada Harga Beli",
        model=st.session_state["linreg_beli_ga"],
        **beli_test
        )
    error_linreg_ga_beli.build()

    error_linreg_ga_jual = MetricsReport(
        title="Hasil Metric Regresi Linier + GA pada Harga Jual",
        model=st.session_state["linreg_jual_ga"],
        **jual_test
        )
    error_linreg_ga_jual.build()