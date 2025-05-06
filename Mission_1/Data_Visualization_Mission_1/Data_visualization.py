import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import streamlit as st

def data_preprocess_visualizaiton(df:pd.DataFrame):
   
    df = df[df['MONAT'] == 'Summe']

    # Group the DataFrame by 'MONATSZAHL' and sum the 'WERT' column
    df_sum_catagory = df.groupby('MONATSZAHL')['WERT'].sum().reset_index()
    # Show the data
    st.subheader("Raw Data")
    st.dataframe(df_sum_catagory)
    # Plotting
    st.title("📊 Total Accidents Data Visualization with Values")
    st.subheader("Bar Chart of Total Accidents")
    fig, ax = plt.subplots()
    bars=ax.bar(df_sum_catagory['MONATSZAHL'], df_sum_catagory['WERT'], color='skyblue')
    # Add text labels on top of the bars
    for bar in bars:
       yval = bar.get_height()
       ax.text(bar.get_x() + bar.get_width()/2, yval + 5, str(yval), ha='center', va='bottom')

    ax.set_xlabel("Categories")
    ax.set_ylabel("Total Accidents")
    ax.set_title("Total Accidents by Category")
    # Show plot in Streamlit
    st.pyplot(fig)
