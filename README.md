
# Time Series Forecasting with TFT Model

This project utilizes the **Temporal Fusion Transformer (TFT)** model for time series forecasting. The dataset consists of monthly traffic accident counts from **January 2000 to December 2020**. The objective is to predict the number of traffic accidents in **January 2021** using a deep learning approach.

## 📁 Project Structure

### Mission_1
- **Data Visualization**: The dataset was visualized to understand trends and seasonality. The result is saved in:
  - `Results for Visualization.png`
- **Model Training and Selection**:
  - The TFT model was trained and saved.
  - After multiple fine-tuning runs, the **three models with the lowest loss values** were selected.
  - These models were evaluated, and their predictions were **averaged** to produce the final forecast.

### Mission_2_Deploy_Google
- The top three models from `Mission_1` were **deployed on Google Cloud** for inference and demonstration purposes.
- The deployment endpoint is: “ https://mission2-783353749415.us-central1.run.app/predict“.
- **NOTE** Only POST can access the prediction result, please following the **Useage Example**
## 🎯 Objective

To demonstrate the effectiveness of the Temporal Fusion Transformer in time series forecasting by predicting traffic accident counts from February 2000 to **January 2021** based on historical data.

## 🔧 Usage Example

To make a prediction for January 2021 using the deployed model, use the following `curl` command:

```bash
curl -X POST "https://mission2-783353749415.us-central1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"year":"2021","month":"01"}'
```
In general you can replace the value of "year" from 2000 to 2021, and the value of "month" from 01 to 12:

```bash
curl -X POST "https://mission2-783353749415.us-central1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"year":"XXXX","month":"XX"}'
```
