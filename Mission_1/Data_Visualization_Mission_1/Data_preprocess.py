import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import streamlit as st

# Function to preprocess the data,get the time period and filter the data with specific conditions
# and save it to a CSV file
def data_preprocess_filter_data(start_year:int, end_year:int, filtered_feature:str, filtered_value:str, file_name:str):
    # Load the data
    data = pd.read_csv('monatszahlen2501_verkehrsunfaelle_27_02_25.csv')

    # Get the time period of data
    data_clip = data[(data['JAHR']<=end_year)& (data['JAHR']>=start_year)]
    # Create a DataFrame with specific columns
    df = pd.DataFrame(data_clip, columns=['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT'])
    # Filter the DataFrame for a specific condition
    df = df[df[filtered_feature] == filtered_value]
     # Save the filtered DataFrame to a CSV file
    df.to_csv(file_name, index=False)
    return df
    #'AUSPRAEGUNG'
    #'insgesamt'
