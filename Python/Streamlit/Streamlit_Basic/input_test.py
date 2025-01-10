import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

emai = st.text_input('Enter Email')
password = st.text_input('Enter Password')
gender = st.selectbox('Select Gender', ['','Male', 'Female', 'Other'])

btn = st.button('Login')

if btn:
    if emai=='naresh@gmail.com' and password=='naresh':
        st.write('Logged In Successfully')
        st.write(gender)
    else:
        st.error('Login Failed')
