import time
import streamlit as st


@st.cache
def gen_algo(size, n_gen, X_train, y_train, cr=0.9, mr=0.5):
    n_feat = X_train.shape[0]
    time.sleep(5)
    return 0