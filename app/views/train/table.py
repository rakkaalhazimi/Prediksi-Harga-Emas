import streamlit as st
import pandas as pd


def get_param_table(params_key, params_lable):
    data = [str(st.session_state.get(req, "-belum ditentukan-")) for req in params_key]
    param_table = pd.DataFrame(data, index=params_lable, columns=["Nilai"])
    return param_table