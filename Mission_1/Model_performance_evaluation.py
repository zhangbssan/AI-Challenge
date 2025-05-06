import pandas as pd
import numpy as np
import data_filter


data_pre = pd.read_csv('Prediction.csv')

df = data_filter.data_preprocess_filter_data(2000, 2020, 'AUSPRAEGUNG', 'insgesamt')
df = df[df['MONAT'] != 'Summe']
df.dropna(inplace=True) 
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
# df.set_index('MONAT', inplace=True)
# sort the data by time
df = df.sort_values(by=["MONAT"])
df =df.reset_index(drop=True)
df['time_idx']=df.index

ground_truth = df['WERT'].values
diff_sqrt = np.sqrt(np.abs(data_pre.values - ground_truth[31:]))
error_mean = np.mean(diff_sqrt)
error_std = np.std(diff_sqrt)

##mean absolute error 5.45 std 1.86
