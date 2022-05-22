import streamlit as st

from config import Config as c
from utils.sessions import set_session
from utils.wrappers import wrap_view
from views.dataset_type.data_loader import load_data, load_custom_data
from views.dataset_type.data_validator import verify_data


def get_ori_data(path):
    ori_df = load_data(path)
    return ori_df

def get_custom_data(user_file):
    custom_df = load_custom_data(user_file)
    valid = verify_data(custom_df)
    if valid:
        return custom_df
    return None

@wrap_view(title="Tipe Dataset")
def view_dataset_type():
    st.write("Pilih salah satu dari tipe dibawah")
    
    ori, custom = "Asli", "Custom"
    option = st.radio(label="Tipe Dataset", options=[ori, custom])
    
    message_ori = "Anda akan menggunakan dataset asli dari server"
    message_custom = """
    Pastikan bahwa data memiliki:
    - 2 kolom dengan nama `HargaBeli` dan `HargaJual` yang berisikan bilangan cacah / bulat
    - 1 kolom dengan nama `Date` yang berisikan tanggal dengan format `DD/MM/YYYY`
    - Jumlah data lebih dari 20
    #
    """

    base_data = get_ori_data(c.DATA_PATH)

    if option == ori:
        st.markdown(message_ori)
        set_session(dataset=base_data, dataset_type=ori)

    else:
        st.markdown(message_custom)
        user_file = st.file_uploader(label="Unggah file .csv, .xlsx", type=["csv", "xlsx"])
        if user_file:
            custom_data = get_custom_data(user_file)
        
        set_session(dataset=custom_data or base_data, dataset_type=custom)