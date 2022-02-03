import streamlit as st
from views import Header, Sidebar


header = Header()
sidebar = Sidebar()

header.build()
sidebar.build()