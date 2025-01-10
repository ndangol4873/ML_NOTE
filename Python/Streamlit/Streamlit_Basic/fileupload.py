import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

dataset =Path("d:\ML_NOTE_DATASET")
dataset


file = st.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    st.dataframe(data.describe())