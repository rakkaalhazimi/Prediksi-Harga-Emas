import streamlit as st
from views import InfoBoard, GAParam, Header

# Header Section
header = Header(
    title="Prediksi Harga Emas", 
    desc="Aplikasi untuk melakukan prediksi pada harga beli atau harga jual emas"
)
header.build()
st.markdown("---")


# Content Section
## Dataset Info
model_info = InfoBoard(
    title="Dataset", 
    desc="Data yang digunakan adalah data harga beli emas dan harga jual emas pada tanggal xx sampai xx")
model_info.build()


## Model Info
model_info = InfoBoard(
    title="Informasi Model", 
    desc="Model yang digunakan adalah regresi linier dan regresi linier dengan algoritma genetika")
model_info.build()


## Genetic Algorithm Parameters
genparam = GAParam()
user_input = genparam.build()