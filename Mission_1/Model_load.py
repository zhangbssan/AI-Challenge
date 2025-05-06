import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
import data_filter
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
# import lightning.pytorch as pl

# app = FastAPI()
model_1 = TemporalFusionTransformer.load_from_checkpoint("AI_MIssion_1_2/Mission_1/checkpoints/tft-epoch=994-train_loss=0.9413.ckpt")
model_2 = TemporalFusionTransformer.load_from_checkpoint("AI_MIssion_1_2/Mission_1/checkpoints/tft-epoch=992-train_loss=0.9608.ckpt")
model_3 = TemporalFusionTransformer.load_from_checkpoint("AI_MIssion_1_2/Mission_1/checkpoints/tft-epoch=954-train_loss=0.9815.ckpt")
# class Item(BaseModel):
#     # time_idx: int
#     year: str
#     month: str
#     # group: "MONATSZAHL"

# # prepare the data
df = data_filter.data_preprocess_filter_data(2000, 2020, 'AUSPRAEGUNG', 'insgesamt')
df = df[df['MONAT'] != 'Summe']
df.dropna(inplace=True) 
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
# df.set_index('MONAT', inplace=True)
# sort the data by time
df = df.sort_values(by=["MONAT"])
df =df.reset_index(drop=True)
df['time_idx']=df.index
list_1= []
# list_2= []
# list_3= []

for i in range(len(df)):
    encoder_data = df[df.time_idx < i+1]
    decoder_data = pd.DataFrame({
    "MONATSZAHL": ["Alkoholunfälle"],
    'AUSPRAEGUNG': ["insgesamt"],
    "JAHR": '0',
    "MONAT": '0',
    "WERT": [0],
    "time_idx": [i + 1]
})
    data = pd.concat([encoder_data,decoder_data], ignore_index=True)
    dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',
        target='WERT',
        group_ids=["MONATSZAHL"],
        max_encoder_length=i+1,
        max_prediction_length=1,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["WERT"],
)
    dataloader = dataset.to_dataloader(train=False, batch_size=1)
    prediction_1 = model_1.predict(dataloader, mode="prediction", return_index=False, return_x=False)
    prediction_2 = model_2.predict(dataloader, mode="prediction", return_index=False, return_x=False)
    prediction_3 = model_3.predict(dataloader, mode="prediction", return_index=False, return_x=False)
    prediction = (prediction_1.item() + prediction_2.item() + prediction_3.item()) / 3
    list_1.append(prediction)

my_list = list_1[:-1] 
my_list = my_list[30:]
df = pd.DataFrame(my_list, columns =['Prediction'])
df.to_csv('Prediction.csv', index=False)