import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
import data_filter
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
# import lightning.pytorch as pl

app = FastAPI()
model = TemporalFusionTransformer.load_from_checkpoint("checkpoints/tft-epoch=90-train_loss=4.6115.ckpt")

class Item(BaseModel):
    # time_idx: int
    year: str
    month: str
    # group: "MONATSZAHL"

# # prepare the data
df = data_filter.data_preprocess_filter_data(2000, 2022, 'AUSPRAEGUNG', 'insgesamt')
df = df[df['MONAT'] != 'Summe']
df.dropna(inplace=True) 
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
# df.set_index('MONAT', inplace=True)
# sort the data by time
df = df.sort_values(by=["MONAT"])
df =df.reset_index(drop=True)
df['time_idx']=df.index
@app.get("/")
def read_root():
    return {"Hello": "TUM"}
    
@app.post("/predict")
def predict(request: Item):
    # json_data = await request.json()
    time_idx_max = (int(request.year)-2000)*12 + int(request.month)-2
    encoder_data = df[df.time_idx <= time_idx_max]
    decoder_data = pd.DataFrame({
    "MONATSZAHL": ["Alkoholunfälle"],
    'AUSPRAEGUNG': ["insgesamt"],
    "JAHR": '0',
    "MONAT": '0',
    "WERT": [0],
    "time_idx": [time_idx_max + 1]
})
    data = pd.concat([encoder_data,decoder_data], ignore_index=True)
    dataset = TimeSeriesDataSet(
    data,
    time_idx='time_idx',
    target='WERT',
    group_ids=["MONATSZAHL"],
    max_encoder_length=time_idx_max+1,
    max_prediction_length=1,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["WERT"]
)
    dataloader = dataset.to_dataloader(train=False, batch_size=1)
    prediction = model.predict(dataloader, mode="prediction", return_index=False, return_x=False)
    final = int(prediction.item())
    return {"prediction": final}
    # return {"prediction":time}




