import os

import streamlit as st
import pandas as pd

@st.cache
def load_data(path):
    return pd.read_csv(path)

def load_custom_data(uploaded):
    fn = uploaded.name.lower()
    ext = os.path.splitext(fn)[-1].lower()

    if ext.endswith(".csv"):
        return pd.read_csv(uploaded)
    elif ext.endswith(".xlsx"):
        return pd.read_excel(uploaded)
    else:
        st.warning("""
        Ekstensi file tidak sesuai, pastikan file berekstensi .csv atau .xlsx, ekstensi saat ini {}
        """.format(ext)
        )
        return False