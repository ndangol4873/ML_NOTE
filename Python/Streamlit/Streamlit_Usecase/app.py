import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


df = pd.read_csv("d:\ML_NOTE_DATASET/startup_funding_cleaned.csv")



st.set_page_config(layout="wide", page_title="Start up Analytics Dashboard development")
st.sidebar.title("Startup Analytics")

option = st.sidebar.selectbox("Categories", ["--Select One --","Over All", "StartUp",'Investor']) 

if option =="--Select One --":
    pass
elif option == "Over All":
    st.title("Overall Analysis")
elif option == "StartUp":
    st.sidebar.selectbox("Select  Start Up", sorted(df['starup'].unique().tolist()))
    btn1 = st.sidebar.button('Find StartUp Details')
    st.title("StartUp Analysis")
else:
    st.sidebar.selectbox("Select", sorted(df['starup'].unique().tolist()))
    btn2 = st.sidebar.button('Find Investor Details')
    st.title("Investor Analysis")