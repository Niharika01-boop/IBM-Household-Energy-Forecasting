*# Time-Series Forecasting of Household Energy Consumption using RNN and LSTM Architectures*



*## 📌 Project Overview*



*This project focuses on forecasting household electricity consumption using recurrent neural network architectures.*



*The project investigates how different recurrent architectures and historical sequence lengths affect forecasting performance.*



*The implemented models are:*



*- Simple RNN*

*- Vanilla LSTM*

*- Stacked LSTM*



*The project performs both:*



*- Single-step forecasting*

*- Multi-step forecasting*



*---*



*## 🎯 Objectives*



*The main objectives are:*



*1. Forecast household energy consumption for the next time step.*

*2. Forecast energy consumption for the next 24 and 48 time steps.*

*3. Compare Simple RNN, Vanilla LSTM, and Stacked LSTM.*

*4. Study the effect of sequence length on forecasting performance.*

*5. Evaluate models using RMSE, MAE, and MAPE.*

*6. Analyze prediction errors using residual analysis.*



*---*



*## 📊 Dataset*



*Dataset:*



*\*\*UCI Individual Household Electric Power Consumption Dataset\*\**



*The dataset contains household electricity measurements collected over time.*



*### Main variables*



*- Global\_active\_power*

*- Global\_reactive\_power*

*- Voltage*

*- Global\_intensity*

*- Sub\_metering\_1*

*- Sub\_metering\_2*

*- Sub\_metering\_3*



*The target variable is:*



*`Global\_active\_power`*



*The raw dataset is not included in this repository because of its large size.*



*---*



*## ⚙️ Data Processing Pipeline*



*The project follows this pipeline:*



*Raw Dataset*  

*↓*  

*Data Cleaning*  

*↓*  

*Datetime Conversion*  

*↓*  

*Missing Value Handling*  

*↓*  

*Hourly Resampling*  

*↓*  

*Exploratory Data Analysis*  

*↓*  

*Cyclical Feature Engineering*  

*↓*  

*Chronological Train/Validation/Test Split*  

*↓*  

*Training-Only Scaling*  

*↓*  

*Sliding Window Creation*  

*↓*  

*RNN/LSTM Training*  

*↓*  

*Evaluation*



*---*



*## 🧠 Feature Engineering*



*The following temporal features were created using cyclical encoding:*



*- Hour of day*

*- Day of week*

*- Month*



*Each temporal variable was represented using sine and cosine transformations.*



*For example:*



*`hour\_sin`*



*`hour\_cos`*



*This allows the model to represent the cyclic nature of time.*



*---*



*## 🤖 Models*



*### Simple RNN*



*A single-layer recurrent neural network.*



*### Vanilla LSTM*



*A single-layer Long Short-Term Memory network.*



*### Stacked LSTM*



*A two-layer LSTM architecture designed to learn deeper temporal representations.*



*---*



*## ⏱️ Sequence Lengths*



*The project investigates:*



*- 24 hours*

*- 72 hours*

*- 168 hours*



*These represent:*



*- 1 day*

*- 3 days*

*- 7 days*



*of historical information.*



*---*



*## 🔮 Forecast Horizons*



*The project evaluates:*



*- 1-step forecasting*

*- 24-step forecasting*

*- 48-step forecasting*



*---*



*## 📏 Evaluation Metrics*



*The models are evaluated using:*



*### RMSE*



*Root Mean Squared Error.*



*### MAE*



*Mean Absolute Error.*



*### MAPE*



*Mean Absolute Percentage Error.*



*MAPE is interpreted carefully because percentage-based errors can become unstable when actual values are close to zero.*



*---*



*## 📈 Results*



*Final experimental results will be added here after completing all experiments.*



*| Model | Sequence Length | Horizon | RMSE | MAE | MAPE |*

*|---|---:|---:|---:|---:|---:|*

*| Simple RNN | 24 | 1 | - | - | - |*

*| Vanilla LSTM | 24 | 1 | - | - | - |*

*| Stacked LSTM | 24 | 1 | - | - | - |*



*---*



*## 📁 Project Structure*



*```text*

*IBM-Household-Energy-Forecasting/*

*│*

*├── data/*

*├── src/*

*├── artifacts/*

*├── requirements.txt*

*├── .gitignore*

*└── README.md*

