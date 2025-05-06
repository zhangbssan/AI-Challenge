import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# Load the data
data = pd.read_csv('monatszahlen_2000-2020.csv')
df = pd.DataFrame(data, columns=['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT'])
df = df[df['MONAT'] != 'Summe']
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']

# Data Normalization
scaler = MinMaxScaler()
WERT_noramalized = scaler.fit_transform(df[['WERT']])
df['WERT_normalized'] = WERT_noramalized

# Convert the data to a time series dataset
# df['MONAT'] = pd.to_datetime(df['MONAT'], format='%Y%m')
df.set_index('MONAT', inplace=True)
# sort the data by time
df = df.sort_values(by=["MONAT"]).reset_index(drop=True)
df['time_idx']=df.index
# Create a time series dataset
dataset = TimeSeriesDataSet(
    df,
    time_idx='time_idx',
    target='WERT',
    group_ids=["MONATSZAHL"],
    max_encoder_length=30,
    max_prediction_length=7,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["WERT"]
)

dataloader = dataset.to_dataloader(train=True, batch_size=64)

model = TemporalFusionTransformer.from_dataset(dataset, learning_rate=1e-3)

# Configure checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="tft-{epoch}-{train_loss:.4f}",
    save_top_k=3,  # Save top 3 best models
    monitor="train_loss",
    mode="min",
)

# Pass to Trainer
# trainer = Trainer(max_epochs=10, accelerator="auto")
trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    max_epochs=100,
    accelerator="cpu",
    enable_model_summary=True,
    limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor d
)

# lightning_model = model.to_lightning_module()
trainer.fit(model, train_dataloaders=dataloader)


