import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os

# Page config
st.set_page_config(
    page_title="SalesGuard - Sales Forecasting & Anomaly Detection",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 SalesGuard: Sales Forecasting & Anomaly Detection")
st.markdown("""
This dashboard provides advanced sales forecasting and anomaly detection using LSTM/GRU networks.
Upload your sales data to get predictions and identify unusual patterns!
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = MinMaxScaler()

# Sidebar
st.sidebar.header("Settings")

# Model Selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["LSTM", "GRU", "Hybrid (LSTM+GRU)"]
)

# Forecasting Settings
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)
lookback_window = st.sidebar.slider("Historical Window (Days)", 7, 90, 30)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type=['csv'])

def create_model(input_shape, model_type="LSTM"):
    model = Sequential()
    
    if model_type == "LSTM":
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
    elif model_type == "GRU":
        model.add(GRU(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(GRU(64))
        model.add(Dropout(0.2))
    else:  # Hybrid
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(GRU(64))
        model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(forecast_horizon))
    
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback:i + lookback + forecast_horizon, 0])
    return np.array(X), np.array(y)

def detect_anomalies(actual, predicted, threshold=2):
    errors = np.abs(actual - predicted)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return errors > (mean_error + threshold * std_error)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Data preview
    st.subheader("Data Preview")
    st.write(df.head())
    
    # Column selection
    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    value_col = st.sidebar.selectbox("Select Value Column", df.columns)
    
    # External variables
    external_vars = st.sidebar.multiselect(
        "Select External Variables (Optional)",
        [col for col in df.columns if col not in [date_col, value_col]]
    )
    
    if st.sidebar.button("Train Model"):
        with st.spinner("Processing data and training model..."):
            # Prepare data
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Scale data
            data = df[[value_col] + external_vars].values
            scaled_data = st.session_state.scaler.fit_transform(data)
            
            # Create sequences
            X, y = create_sequences(scaled_data, lookback_window)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Create and train model
            st.session_state.model = create_model(
                (lookback_window, 1 + len(external_vars)),
                model_type
            )
            
            history = st.session_state.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            st.success("Model trained successfully!")
    
    if st.session_state.model is not None:
        st.subheader("Forecasting & Anomaly Detection")
        
        # Make predictions
        data = df[[value_col] + external_vars].values
        scaled_data = st.session_state.scaler.transform(data)
        X, _ = create_sequences(scaled_data, lookback_window)
        predictions = st.session_state.model.predict(X)
        
        # Inverse transform predictions
        pred_data = np.zeros((len(predictions), data.shape[1]))
        pred_data[:, 0] = predictions.flatten()
        predictions_unscaled = st.session_state.scaler.inverse_transform(pred_data)[:, 0]
        
        # Detect anomalies
        anomalies = detect_anomalies(
            df[value_col].values[lookback_window:],
            predictions_unscaled[:-forecast_horizon]
        )
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Historical data with anomalies
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=df[date_col][lookback_window:],
                y=df[value_col][lookback_window:],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Anomalies
            anomaly_dates = df[date_col][lookback_window:][anomalies]
            anomaly_values = df[value_col][lookback_window:][anomalies]
            
            fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(title="Sales Time Series with Anomalies")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Future forecast
            future_dates = pd.date_range(
                df[date_col].iloc[-1],
                periods=forecast_horizon + 1,
                freq='D'
            )[1:]
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df[date_col][-30:],
                y=df[value_col][-30:],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions_unscaled[-1],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(title="Sales Forecast")
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mse = np.mean((df[value_col].values[lookback_window:] - predictions_unscaled[:-forecast_horizon])**2)
            st.metric("Mean Squared Error", f"{mse:.2f}")
        
        with col2:
            mae = np.mean(np.abs(df[value_col].values[lookback_window:] - predictions_unscaled[:-forecast_horizon]))
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        
        with col3:
            anomaly_percentage = (anomalies.sum() / len(anomalies)) * 100
            st.metric("Anomaly Percentage", f"{anomaly_percentage:.1f}%")
        
        # External Variables Impact
        if external_vars:
            st.subheader("External Variables Impact")
            correlation_data = df[[value_col] + external_vars].corr()[value_col].sort_values()
            
            fig = px.bar(
                x=correlation_data.index,
                y=correlation_data.values,
                title="Correlation with Sales"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to get started!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using TensorFlow and Streamlit") 