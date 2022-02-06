import streamlit as st
import pandas as pd

@st.cache
def load_data():
    return pd.read_csv("data/dataset_full.csv")