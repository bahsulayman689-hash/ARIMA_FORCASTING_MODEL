# ARIMA Time Series Forecasting Dashboard

An interactive web application built with Streamlit to clean time series data, evaluate stationarity characteristics, optimize model hyperparameters, perform train-test backtesting, and project out-of-sample future forecasts.
-**paste here your live demo**[]
## Features
- **Data Preprocessing**: Upload custom CSV files, select target features dynamically, and parse historical timestamp indices.
- **Stationarity Diagnostic Badges**: Interactive Augmented Dickey-Fuller (ADF) tests evaluating original data alongside differenced variations.
- **Correlation Diagnostics**: Real-time rendering pipelines for tracking Autocorrelation (ACF) and Partial Autocorrelation (PACF) properties.
- **Model Tuning Alternatives**: Toggle between direct manual sliders or automated loop-driven Grid Search algorithms mapping AIC values.
- **Performance Evaluation**: Embedded backtesting splitting modules detailing Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) profiles.
- **Dynamic Future Horizons**: User-controlled lookback context dimensions mapping forecast confidence segments.

## Getting Started

### 1. Initialize Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows shell configurations use: venv\Scripts\activate
```

### 2. Install Package Dependencies
Install the required application library components via pip environment tooling:
```bash
pip install -r requirements.txt
```

### 3. Launch the Server Dashboard Application
Execute the deployment rendering stream locally inside your terminal space:
```bash
streamlit run arima_app.py
```
