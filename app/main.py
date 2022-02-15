import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from views import *
from styles import css_style
from src.data import load_data
from src.models import gen_algo, combine_predictions, prediction_date_based
from src.pre import preprocess_data

# Muat style CSS
css_style()

# Tampilan Antar Muka
view_home()
view_tutorial()
view_dataset_type()
view_parameter()
view_train()
view_result()
view_comparison()
view_charts()
view_predict_period()
view_predict_date()