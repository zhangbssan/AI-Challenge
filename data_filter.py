import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt

def data_preprocess_filter_data(start_year:int, end_year:int, filtered_feature:str, filtered_value:str):
    # Load the data
    data = pd.read_csv('monatszahlen2501_verkehrsunfaelle_27_02_25.csv')

    # Get the time period of data
    data_clip = data[(data['JAHR']<=end_year)& (data['JAHR']>=start_year)]
    # Create a DataFrame with specific columns
    df = pd.DataFrame(data_clip, columns=['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT'])
    # Filter the DataFrame for a specific condition
    df = df[df[filtered_feature] == filtered_value]
    #  # Save the filtered DataFrame to a CSV file
    # df.to_csv(file_name, index=False)
    return df