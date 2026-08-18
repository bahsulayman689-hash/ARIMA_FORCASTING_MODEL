import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import itertools

# Hide unnecessary warnings
warnings.filterwarnings("ignore")

# Set up page configurations
st.set_page_config(
    page_title="ARIMA Time Series Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ARIMA Time Series Forecasting Dashboard")
st.markdown("""
This application allows you to upload a time series dataset, evaluate its stationarity, 
automatically discover or manually select optimal ARIMA parameters, validate the model's accuracy, and generate future forecasts.
""")

# Sidebar controls for configuration
st.sidebar.header("🛠️ Model Configuration")

# File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload your Time Series CSV", type=["csv"])

@st.cache_data
def load_sample_data():
    """Generates synthetic temperature data matching the structure of daily-min-temperatures to ensure the app is immediately runnable."""
    dates = pd.date_range(start="1990-01-01", periods=365, freq="D")
    # Simulate a seasonal pattern with some random noise
    base_temp = 15 + 5 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 2, size=len(dates))
    df = pd.DataFrame({"Date": dates, "Temp": base_temp + noise})
    return df

# Handle Data Ingestion
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()
else:
    st.sidebar.info("Using sample simulated dataset. Upload your own CSV to analyze custom data.")
    data = load_sample_data()

# Column Selection
st.subheader("📊 Dataset Preview & Configuration")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("First few rows of the dataset:")
    st.dataframe(data.head())
    
    columns = data.columns.tolist()
    date_col = st.selectbox("Select Date/Index Column", columns, index=0)
    target_col = st.selectbox("Select Value/Target Column", columns, index=1 if len(columns) > 1 else 0)

# Pre-processing Data
try:
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    data = data.dropna(subset=[target_col])
    
    # Ensure index is datetime if parsing is successful
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    if not data[date_col].isnull().all():
        data = data.set_index(date_col)
    else:
        st.warning("Date column formatting could not be fully parsed into Datetime. Using standard numeric indexing instead.")
    
    ts = data[target_col]
except Exception as e:
    st.error(f"Error processing columns: {e}")
    st.stop()

with col2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values, color='#1f77b4', label=target_col)
    ax.set_title(f"Time Series Plot: {target_col}")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# Stationarity Testing Section
st.markdown("---")
st.subheader("🔍 Stationarity Analysis (ADF Test)")

col_adf1, col_adf2 = st.columns(2)

with col_adf1:
    st.markdown("**Original Series Test**")
    result = adfuller(ts)
    st.metric(label="ADF Statistic", value=f"{result[0]:.4f}")
    st.metric(label="p-value", value=f"{result[1]:.4f}")
    
    if result[1] > 0.05:
        st.error("❌ Series is non-stationary; differencing is recommended.")
        need_diff = True
    else:
        st.success("✅ Series is stationary; no differencing needed.")
        need_diff = False

with col_adf2:
    st.markdown("**First-Order Differenced Series Test**")
    ts_diff = ts.diff().dropna()
    result_diff = adfuller(ts_diff)
    st.metric(label="ADF Statistic (Differenced)", value=f"{result_diff[0]:.4f}")
    st.metric(label="p-value (Differenced)", value=f"{result_diff[1]:.4f}")
    
    if result_diff[1] > 0.05:
        st.warning("⚠️ Differenced series is still non-stationary. Higher order differences may be required.")
    else:
        st.success("✅ Differenced series is stationary.")

# Visualizing Diagnostics (ACF / PACF)
st.subheader("📉 Diagnostic Correlation Plots")
show_plots = st.checkbox("Show ACF and PACF Plots for the Differenced Series", value=True)

if show_plots:
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        fig_acf, ax_acf = plt.subplots(figsize=(6, 3.5))
        plot_acf(ts_diff, ax=ax_acf)
        ax_acf.set_title("Autocorrelation (ACF)")
        st.pyplot(fig_acf)
    with col_plot2:
        fig_pacf, ax_pacf = plt.subplots(figsize=(6, 3.5))
        plot_pacf(ts_diff, ax=ax_pacf)
        ax_pacf.set_title("Partial Autocorrelation (PACF)")
        st.pyplot(fig_pacf)

# Model Fitting Settings
st.markdown("---")
st.subheader("🤖 ARIMA Optimization & Tuning")

tuning_method = st.radio("Choose hyperparameter tuning method:", ["Manual Selection", "Auto Grid Search Optimization"])

best_order = (1, 1, 1) # Default fallback

if tuning_method == "Auto Grid Search Optimization":
    st.markdown("Hyperparameter grid evaluation spaces:")
    max_p = st.slider("Max Autoregressive term (p)", min_value=1, max_value=4, value=2)
    max_d = st.slider("Max Differencing order (d)", min_value=0, max_value=2, value=1)
    max_q = st.slider("Max Moving Average term (q)", min_value=1, max_value=4, value=2)
    
    if st.button("🚀 Run Grid Search Optimization"):
        p_range = range(0, max_p + 1)
        d_range = range(0, max_d + 1)
        q_range = range(0, max_q + 1)
        pdq = list(itertools.product(p_range, d_range, q_range))
        
        best_aic = np.inf
        
        progress_bar = st.progress(0)
        total_iterations = len(pdq)
        
        for idx, order in enumerate(pdq):
            try:
                model = ARIMA(ts, order=order)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
            except:
                continue
            progress_bar.progress((idx + 1) / total_iterations)
            
        st.success(f"Best ARIMA order identified: **{best_order}** with an AIC score of: **{best_aic:.2f}**")
        st.session_state['optimal_order'] = best_order
else:
    # Manual Configuration sliders
    p_param = st.sidebar.slider("AR order (p)", 0, 5, 1)
    d_param = st.sidebar.slider("Differencing degree (d)", 0, 2, 1)
    q_param = st.sidebar.slider("MA order (q)", 0, 5, 1)
    best_order = (p_param, d_param, q_param)
    st.session_state['optimal_order'] = best_order

# Retrieve order from session state tracking if present
final_order = st.session_state.get('optimal_order', best_order)
st.info(f"Fitting model with structural configuration parameters: **ARIMA{final_order}**")

# Training Final Model Instance
try:
    final_model = ARIMA(ts, order=final_order)
    fitted_results = final_model.fit()
    
    # Model Summary Expandable Section
    with st.expander("📄 View Full Mathematical Model Summary"):
        st.text(fitted_results.summary().as_text())
except Exception as e:
    st.error(f"Failed to fit ARIMA model with the specified order details: {e}")
    st.stop()

# Lookback settings shared between evaluation and forecasting
lookback = st.sidebar.slider("Historical intervals context window to display on charts:", min_value=10, max_value=100, value=30)

# Validation Section (Train-Test Split)
st.markdown("---")
st.subheader("📏 Model Evaluation & Accuracy Metrics")

enable_validation = st.checkbox("Enable Train-Test Split Validation", value=True)

if enable_validation:
    test_size = st.slider("Select test set size (number of observations to hold out):", min_value=3, max_value=min(60, len(ts)//4), value=min(15, len(ts)//10))
    
    train_ts = ts[:-test_size]
    test_ts = ts[-test_size:]
    
    try:
        val_model = ARIMA(train_ts, order=final_order)
        val_results = val_model.fit()
        val_forecast = val_results.forecast(steps=test_size)
        
        # Calculate metrics using numpy to prevent extra scikit-learn dependencies
        mae = np.mean(np.abs(test_ts.values - val_forecast.values))
        rmse = np.sqrt(np.mean((test_ts.values - val_forecast.values)**2))
        mape = np.mean(np.abs((test_ts.values - val_forecast.values) / test_ts.values)) * 100
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
        col_m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
        col_m3.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
        
        # Plot validation results
        fig_val, ax_val = plt.subplots(figsize=(10, 4))
        ax_val.plot(train_ts[-lookback:].index, train_ts[-lookback:].values, label='Training Context (Actual)', color='blue', marker='o')
        ax_val.plot(test_ts.index, test_ts.values, label='Holdout Test (Actual)', color='green', marker='s')
        ax_val.plot(test_ts.index, val_forecast.values, label='Validation Prediction', color='orange', linestyle='--', marker='x')
        ax_val.set_title("Train / Test Backtesting Validation Alignment", fontsize=14)
        ax_val.set_xlabel("Timeline Index", fontsize=12)
        ax_val.set_ylabel(target_col, fontsize=12)
        ax_val.legend()
        ax_val.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_val)
        
    except Exception as e:
        st.error(f"Validation calculations failed with current structural configuration layout: {e}")

# Forecasting Dashboard View
st.markdown("---")
st.subheader("🔮 Forecasting Projections")

steps = st.slider("Steps / Intervals to forecast into future:", min_value=5, max_value=60, value=10)

if st.button("🎯 Generate Out-of-Sample Forecast"):
    try:
        forecast_values = fitted_results.forecast(steps=steps)
        
        # Match alignment mechanics depending on type of pandas indexing structure used
        fig_fc, ax_fc = plt.subplots(figsize=(10, 5))
        
        # Display slicing values safely
        hist_ts = ts[-lookback:]
        
        ax_fc.plot(hist_ts.index, hist_ts.values, label='Actual Historical Context', color='blue', linestyle='-', marker='o')
        ax_fc.plot(forecast_values.index, forecast_values.values, label='Out-of-Sample Forecast', color='red', linestyle='--', marker='x')
        
        ax_fc.set_title('ARIMA Predictive Model Inference Plot', fontsize=14)
        ax_fc.set_xlabel('Timeline Index', fontsize=12)
        ax_fc.set_ylabel(target_col, fontsize=12)
        ax_fc.legend()
        ax_fc.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig_fc)
        
        # Display projections in structured tabular form
        st.markdown("**Tabular Projection Schedule Data Output:**")
        fc_df = pd.DataFrame({"Forecasted Projection Point": forecast_values.values}, index=forecast_values.index)
        st.dataframe(fc_df.transpose())
        
    except Exception as e:
        st.error(f"Inference processing calculation engine error: {e}")
