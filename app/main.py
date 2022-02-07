import streamlit as st
from sklearn.linear_model import LinearRegression
from views import InfoBoard, InfoBoardWithButton, GAParam, PreParam, Header
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
st.session_state["is_train"] = is_train

if st.session_state.get("size") and is_train:
    # Latih Regresi Linier
    st.write("Melatih regresi linier...")
    linreg_beli = LinearRegression().fit(beli_train["X_train"], beli_train["y_train"])
    linreg_jual = LinearRegression().fit(jual_train["X_train"], jual_train["y_train"])
    st.write("Selesai")

    # Latih Regresi Linier dengan GA
    st.markdown("#")
    st.write("Melatih regresi linier dengan GA...")
    ga_input = {
        "size": st.session_state.get("size"),
        "n_gen": st.session_state.get("n_gen"),
        "cr": st.session_state.get("cr"),
        "mr": st.session_state.get("mr"),
    }
    population_beli, fitness_beli, linreg_beli_ga = gen_algo(**ga_input, **beli_train)
    population_jual, fitness_jual, linreg_jual_ga = gen_algo(**ga_input, **beli_train)
    st.write("Selesai")

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