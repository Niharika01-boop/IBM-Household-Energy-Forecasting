🏠 Time-Series Forecasting of Household Energy Consumption

<p align="center">
  <b>RNN • LSTM • Stacked LSTM • Multi-Horizon Forecasting</b>
</p>

A complete deep-learning pipeline for forecasting household electricity consumption from the UCI Individual Household Electric Power Consumption dataset.

The project covers the workflow end-to-end: data cleaning, hourly resampling, exploratory analysis, cyclical feature engineering, chronological splitting, train-only scaling, sliding-window sequence generation, recurrent neural-network training, multi-horizon forecasting, inverse scaling, evaluation, visualization, and residual analysis.

📌 Project Overview

Problem

Household electricity demand is a time-dependent signal with daily, weekly, and longer-term patterns. The objective is to learn these temporal dependencies and forecast future Global_active_power.

Forecasting Tasks

The project evaluates:

Single-step forecasting: next 1 hour

Multi-step forecasting: next 24 hours

Multi-step forecasting: next 48 hours

Model Architectures

Three recurrent architectures are compared:

Simple RNN

Vanilla LSTM

Stacked LSTM

Historical Sequence Lengths

Experiments use:

T = 24 → previous 24 hours

T = 72 → previous 72 hours

T = 168 → previous 7 days

This gives 27 model/sequence/horizon experiments across the three architectures.

🎯 Objectives

Forecast household electricity consumption one step ahead.

Forecast the next 24 and 48 hourly observations.

Compare Simple RNN, Vanilla LSTM, and Stacked LSTM.

Study how historical sequence length affects forecasting.

Evaluate models using RMSE, MAE, and MAPE.

Compare actual vs. predicted consumption.

Analyze residual distributions.

Preserve chronological ordering to avoid time-series leakage.

Build a reproducible PyTorch training and evaluation pipeline.

📊 Dataset

This project uses the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository.

The original dataset contains 2,075,259 measurements collected at one-minute intervals over almost four years, with seven main numerical electrical measurements plus date/time fields.

Main variables

Feature

Description

Global_active_power

Household global active power

Global_reactive_power

Household global reactive power

Voltage

Average voltage

Global_intensity

Current intensity

Sub_metering_1

Kitchen-related sub-metering

Sub_metering_2

Laundry-related sub-metering

Sub_metering_3

Water-heater / air-conditioner-related sub-metering

Target: Global_active_power

Dataset source:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

🔄 End-to-End Pipeline

Raw UCI Dataset
      │
      ▼
Data Loading & Cleaning
      │
      ▼
Missing-Value Handling
      │
      ▼
Hourly Resampling
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Cyclical Feature Engineering
(hour / day-of-week / month)
      │
      ▼
Chronological Train / Validation / Test Split
      │
      ▼
Train-Only Feature & Target Scaling
      │
      ▼
Sliding-Window Sequence Generation
      │
      ├──────────────┐
      ▼              ▼
   RNN            LSTM
                     │
                     ▼
                Stacked LSTM
      │              │
      └──────┬───────┘
             ▼
       Multi-Horizon Forecasting
             │
             ▼
       Inverse Transformation
             │
             ▼
      RMSE / MAE / MAPE
             │
             ▼
  Plots + Residual Analysis + Results

🧹 Data Processing

The raw minute-level data is:

Parsed with the original date/time information.

Converted to numeric features.

Interpolated for missing time-series values.

Resampled from minute-level observations to hourly averages.

Prepared for feature engineering and forecasting.

The resulting hourly dataset contains:

34,589 hourly observations

7 original numerical electrical features

🧠 Feature Engineering

The forecasting input contains the original electrical variables plus cyclical time features.

Original features

Global_active_power

Global_reactive_power

Voltage

Global_intensity

Sub_metering_1

Sub_metering_2

Sub_metering_3

Added cyclical features

hour_sin

hour_cos

dow_sin

dow_cos

month_sin

month_cos

Total model input features: 13

Cyclical encoding allows periodic time information to be represented without introducing artificial discontinuities between the beginning and end of a cycle.

📐 Data Split & Scaling

The data is split chronologically:

Split

Proportion

Training

70%

Validation

15%

Test

15%

No random shuffling is used for the time-series split.

Feature and target scalers are fitted only on the training data and then applied to validation/test data. This prevents information from future periods leaking into model training.

🧩 Sequence Construction

Sliding windows are used to transform the time series into supervised learning samples.

For a sequence length T and forecast horizon H:

Input:
[t-T+1, ..., t]

Target:
[t+1, ..., t+H]

Experiments:

Sequence Length

Forecast Horizon

24

1, 24, 48

72

1, 24, 48

168

1, 24, 48

🤖 Model Architectures

Simple RNN

A recurrent neural network that processes the historical sequence and learns temporal dependencies.

Vanilla LSTM

A single-layer Long Short-Term Memory network designed to capture longer-term dependencies using gated memory.

Stacked LSTM

A multi-layer LSTM architecture that learns hierarchical temporal representations.

All models use the same general experimental framework so that architecture, sequence length, and forecast horizon can be compared consistently.

⚙️ Training Configuration

The training pipeline uses PyTorch.

Typical configuration:

Optimizer: AdamW

Learning rate: 0.001

Loss: Mean Squared Error (MSE)

Batch size: 64

Hidden size: 64

Gradient clipping: 1.0

Maximum epochs: 30

Early stopping patience: 5

📏 Evaluation Metrics

RMSE

Measures the square root of mean squared prediction error.

Lower RMSE indicates smaller prediction errors with greater sensitivity to large errors.

MAE

Measures the average absolute prediction error.

Lower MAE indicates better average accuracy.

MAPE

Measures error relative to the actual value.

Because percentage error can become unstable when actual values are close to zero, RMSE and MAE are also considered when interpreting results.

📈 Results & Analysis

The repository contains the complete experiment outputs.

Results table

artifacts/results/results_table.csv

Prediction files

Each experiment stores actual and predicted values as .npz files:

artifacts/results/

Generated comparisons

The project includes:

RMSE comparison plots

MAE comparison plots

Horizon-specific comparisons

Overall model comparisons

Residual distribution plots

Actual vs. predicted analysis

These outputs make it possible to study the effects of:

Model architecture

Historical sequence length

Forecast horizon

📁 Project Structure

IBM-Household-Energy-Forecasting/
│
├── artifacts/
│   ├── evaluation/
│   ├── models/
│   ├── results/
│   └── *.png / *.csv
│
├── data/
│   ├── household_power_consumption.txt
│   ├── household_power_hourly.csv
│   ├── household_power_features.csv
│   ├── scaled/
│   └── sequences/
│
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── resample_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── split_and_scale.py
│   ├── create_sequences.py
│   ├── create_all_sequences.py
│   ├── create_all_horizon_sequences.py
│   ├── dataloader.py
│   ├── models.py
│   ├── train.py
│   ├── train_lstm.py
│   ├── train_stacked_lstm.py
│   ├── train_experiment.py
│   ├── evaluate.py
│   ├── evaluate_lstm.py
│   ├── evaluate_stacked_lstm.py
│   ├── evaluate_experiment.py
│   ├── compare_models.py
│   ├── make_results_table.py
│   ├── plot_results.py
│   └── plot_residuals.py
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt

🚀 How to Run

1. Clone the repository

git clone https://github.com/Niharika01-boop/IBM-Household-Energy-Forecasting.git
cd IBM-Household-Energy-Forecasting

2. Install dependencies

pip install -r requirements.txt

3. Run preprocessing

python src/clean_data.py
python src/resample_data.py
python src/eda.py
python src/feature_engineering.py
python src/split_and_scale.py

4. Create forecasting sequences

For the full experiment matrix:

python src/create_all_horizon_sequences.py

5. Train an experiment

Example:

python src/train_experiment.py --model lstm --T 72 --H 48

6. Evaluate an experiment

python src/evaluate_experiment.py --model lstm --T 72 --H 48

7. Generate result comparisons

python src/make_results_table.py
python src/plot_results.py
python src/plot_residuals.py

🔬 Reproducibility

The repository contains the complete project artifacts used during experimentation, including:

Raw dataset

Processed datasets

Scalers

Sequence arrays

Trained model checkpoints

Prediction outputs

Evaluation results

Visualization outputs

Source code

Large files are managed using Git LFS.

🛠️ Technology Stack

Python

PyTorch

Pandas

NumPy

Scikit-learn

Matplotlib

Joblib

Git / GitHub

Git LFS

💡 Key Learning Outcomes

This project demonstrates practical experience with:

Time-series preprocessing

Missing-value handling

Temporal feature engineering

Data leakage prevention

Sliding-window forecasting

Recurrent neural networks

LSTM architectures

Multi-step forecasting

Model evaluation

Error and residual analysis

Experiment management

Reproducible ML workflows

Git and Git LFS for large ML artifacts

🔮 Future Improvements

Potential extensions include:

Hyperparameter optimization

Attention-based recurrent models

Temporal Convolutional Networks

Transformer-based forecasting

Probabilistic forecasting

Longer forecasting horizons

Automated experiment tracking

Deployment through a FastAPI service

Interactive forecasting dashboard

📚 Dataset Citation

Hebrail, G. & Berard, A. (2006).
Individual Household Electric Power Consumption.
UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C58K54

👩‍💻 Project

GitHub:
https://github.com/Niharika01-boop/IBM-Household-Energy-Forecasting

Repository: Public and includes the complete project artifacts managed with Git LFS.
