*# Time-Series Forecasting of Household Energy Consumption using RNN and LSTM*



*A deep learning project for forecasting household electricity consumption using recurrent neural network architectures, including Simple RNN, Vanilla LSTM, and Stacked LSTM.*



*## 📌 Project Overview*



*This project investigates time-series forecasting of household electricity consumption using recurrent neural networks.*



*The primary target variable is:*



*`Global\_active\_power`*



*The project evaluates both short-term and long-horizon forecasting using different historical sequence lengths and forecasting horizons.*



*---*



*## 🎯 Objectives*



*The project aims to:*



*- Forecast household electricity consumption one step ahead.*

*- Forecast the next 24 hours.*

*- Forecast the next 48 hours.*

*- Compare Simple RNN, Vanilla LSTM, and Stacked LSTM architectures.*

*- Study the effect of sequence length on forecasting performance.*

*- Evaluate models using RMSE, MAE, and MAPE.*

*- Analyze prediction errors using residual distributions.*

*- Compare actual and predicted energy consumption.*



*---*



*## 📊 Dataset*



*The project uses the \*\*UCI Household Electric Power Consumption Dataset\*\*.*



*The dataset contains household electricity measurements recorded at one-minute intervals.*



*### Main variables*



*- Global\_active\_power*

*- Global\_reactive\_power*

*- Voltage*

*- Global\_intensity*

*- Sub\_metering\_1*

*- Sub\_metering\_2*

*- Sub\_metering\_3*



*The data was resampled from minute-level observations to hourly observations.*



*### Target*



*`Global\_active\_power`*



*---*



*## 🔄 Project Pipeline*



*```text*

*Raw Dataset*

&#x20;    *↓*

*Data Cleaning*

&#x20;    *↓*

*Missing Value Handling*

&#x20;    *↓*

*Hourly Resampling*

&#x20;    *↓*

*Exploratory Data Analysis*

&#x20;    *↓*

*Feature Engineering*

&#x20;    *↓*

*Chronological Train / Validation / Test Split*

&#x20;    *↓*

*Train-only Feature Scaling*

&#x20;    *↓*

*Sliding Window Sequence Creation*

&#x20;    *↓*

*RNN / LSTM / Stacked LSTM*

&#x20;    *↓*

*Forecasting*

&#x20;    *↓*

*Inverse Scaling*

&#x20;    *↓*

*RMSE / MAE / MAPE*

&#x20;    *↓*

*Visualization \& Residual Analysis*

