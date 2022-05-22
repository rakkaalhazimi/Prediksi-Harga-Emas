import streamlit as st

def set_session(**kwargs):
    for key, value in kwargs.items():
        st.session_state[key] = value
