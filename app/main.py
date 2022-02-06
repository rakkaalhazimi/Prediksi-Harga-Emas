import streamlit as st
from views import InfoBoard, GAParam, PreParam, Header
from src.data import load_data
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
user_input = test_param.build()

# Pre - Praproses Data
df_use = preprocess_data


# View - Parameter Algoritma Genetika
gen_param = GAParam()
user_input = gen_param.build()