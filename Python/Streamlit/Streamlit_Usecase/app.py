import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path 


st.set_page_config(layout="wide", page_title="Start up Analytics Dashboard development")

dataset =Path("d:\ML_NOTE_DATASET")
df = pd.read_csv(f"{dataset}/startup_funding_cleaned.csv")
# df['date'] = pd.to_datetime(df['date'],errors='coerce')
# df['month'] = df['date'].dt.month
# df['year'] = df['date'].dt.year
# st.dataframe(df)


st.sidebar.title("Startup Analytics")


## Load investor details function 
def load_investor_details(investor):
    st.title(investor)

    ## Most recent Investment of the Investors
    most_recent_investment = df.loc[df['investors'].str.contains(investor)].head()[['date','starup','vertical','city','round','amount']].reset_index(drop=True)
    most_recent_investment.index = most_recent_investment.index+1 ## Index from 1
    st.header('Most Recent Investments')
    st.dataframe(most_recent_investment)

    ##Biggest Investment
    big_invesment = df.loc[df['investors'].str.contains(investor)].groupby('starup')['amount'].sum().sort_values(ascending=False).head().reset_index()
    big_invesment.index = big_invesment.index+1
    st.header('Biggest Investments Details')
    # st.dataframe(big_invetment)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Investments Table')
        st.dataframe(big_invesment)

    with col2:
        st.subheader('Investments Bar Chart')
        fig, ax = plt.subplots() 
        ax.bar(big_invesment['starup'], big_invesment['amount'])
        ax.set_xticklabels(big_invesment['starup'],rotation= 90)
        st.pyplot(fig)


    def create_pie_chart(data, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(data, labels=data.index, autopct="%0.01f%%")
        ax.set_title(title)
        return fig
    
    verical_series = df[df['investors'].str.contains(investor)].groupby('vertical')['amount'].sum()
    col11, col22, col33 = st.columns(3)
    with col11:
        st.subheader('Sectors invested in')
        fig1 = create_pie_chart(verical_series, 'Sectors invested in')
        st.pyplot(fig1)

    
    round_series = df[df['investors'].str.contains(investor)].groupby('round')['amount'].sum()
    with col22:
        st.subheader('Rounds invested in')
        fig2 = create_pie_chart(round_series, 'Rounds invested in')
        st.pyplot(fig2)


    city_series = df[df['investors'].str.contains(investor)].groupby('city')['amount'].sum()
    with col33:
        st.subheader('Cities invested in')
        fig3 = create_pie_chart(city_series, 'Cities invested in')
        st.pyplot(fig3)






## Categories Drop Down
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
    selected_investor = st.sidebar.selectbox("Select Investor", sorted(set(df["investors"].str.split(',').sum())))
    btn2 = st.sidebar.button('Find Investor Details')
    if btn2:
        load_investor_details(selected_investor)

   