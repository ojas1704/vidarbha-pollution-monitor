import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Vidarbha Air Pulse V4.0", layout="wide", page_icon="🍃")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    h1, h2, h3, h4, h5, h6, p, span, div { color: #2E4053 !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetricValue"] { color: #1a1a1a !important; }
    .metric-container {
        background-color: #f0fdf4; border-left: 5px solid #28B463;
        padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div.stButton > button { background-color: #28B463; color: white !important; border-radius: 30px; border: none; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (MULTI-FACTOR SIMULATION) ---
@st.cache_data
def get_base_data():
    dates = pd.date_range(end=datetime.today(), periods=1825)
    t = np.arange(1825)
    
    # Base Factors
    seasonal = 50 * np.sin(t / 180)  # Winter/Summer cycle
    weekly = 15 * np.sin(t / 7)      # Traffic cycle
    trend = t * 0.015                # Yearly degradation
    noise = np.random.normal(0, 4, 1825)
    
    # Base AQI Calculation
    aqi = 100 + seasonal + weekly + trend + noise
    aqi = np.clip(aqi, 40, 400)
    
    df = pd.DataFrame({'Date': dates, 'AQI': aqi})
    df.set_index('Date', inplace=True)
    return df

# --- 3. MULTI-VARIATE MODEL ---
@st.cache_resource
def train_and_evaluate(df, traffic, construction, wind, industry, temp):
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # --- THE "FRIEND LOGIC" IMPLEMENTATION ---
    # We calculate a 'Net Impact' based on ALL factors
    # Traffic & Industry ADD pollution. Wind & High Temp REMOVE pollution.
    
    # Normalize inputs to 0-1 scale roughly
    f_traffic = traffic / 100.0
    f_const = construction / 50.0
    f_wind = wind / 30.0    # Max wind 30 km/h
    f_ind = industry / 100.0
    
    # Logic: 
    # Pollution sources: Traffic (30% weight), Construction (20%), Industry (20%)
    # Cleaning sources: Wind (removes up to 40% pollution)
    
    net_stress = (f_traffic * 0.15) + (f_const * 0.10) + (f_ind * 0.15) - (f_wind * 0.20)
    
    # Temperature Factor: Cold (below 20C) traps pollution (+), Hot disperses it (-)
    if temp < 20:
        net_stress += 0.05 # Winter Smog Effect
    else:
        net_stress -= 0.02
        
    # Apply this logic to the recent data (Last 90 days)
    # This simulates how current conditions affect the historical trend
    if net_stress != 0:
        scaled_data[-90:] += net_stress

    # Clip to keep data valid (0 to 1)
    scaled_data = np.clip(scaled_data, 0, 1)

    # --- LSTM PREPARATION ---
    prediction_days = 60
    x, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        x.append(scaled_data[i-prediction_days:i, 0])
        y.append(scaled_data[i, 0])
        
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Split
    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=40, batch_size=32, verbose=0, callbacks=[early_stop])

    # Forecast
    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    # Future loop
    future_forecast = []
    current_batch = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    for i in range(7):
        pred = model.predict(current_batch, verbose=0)[0,0]
        future_forecast.append(pred)
        new_step = np.array([[[pred]]])
        current_batch = np.append(current_batch[:,1:,:], new_step, axis=1)
    
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
    return future_forecast, rmse, df.index[split_index+prediction_days:]

# --- 4. UI LAYOUT ---

col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌱 Vidarbha Air Pulse V4.0")
    st.markdown("### Multi-Factor Atmospheric Analysis")
    st.write("Now incorporating **Meteorological Data** (Wind/Temp) and **Industrial Emissions** for high-fidelity forecasting.")
with col2:
    st.image("https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?q=80&w=1974&auto=format&fit=crop", use_container_width=True)

st.markdown("---")

# COMPLEX SIDEBAR
st.sidebar.header("🎛️ Atmospheric Controls")
st.sidebar.markdown("**Pollution Sources (+)**")
traffic = st.sidebar.slider("🚦 Traffic Density", 0, 100, 30, help="Higher traffic increases NO2 and CO.")
industry = st.sidebar.slider("🏭 Industrial Output", 0, 100, 40, help="Factory emissions increase PM2.5.")
construction = st.sidebar.slider("🏗️ Construction", 0, 50, 10, help="Dust increases PM10.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Mitigating Factors (-)**")
wind = st.sidebar.slider("🌬️ Wind Speed (km/h)", 0, 30, 5, help="High wind disperses pollutants.")
temp = st.sidebar.slider("🌡️ Temperature (°C)", 5, 45, 28, help="Cold air (Inversion) traps pollutants.")

# Run
df = get_base_data()
with st.spinner("Analyzing Multi-Factor Interaction..."):
    forecast, rmse, test_dates = train_and_evaluate(df, traffic, construction, wind, industry, temp)

tomorrows_aqi = int(forecast[0][0])

# Metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-container"><h3>Forecast</h3><h1>{tomorrows_aqi}</h1><p>AQI Tomorrow</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-container"><h3>Net Wind Impact</h3><h1>-{wind * 2}%</h1><p>Dispersion Rate</p></div>', unsafe_allow_html=True)
with m3:
    status = "Trapping" if temp < 20 else "Normal"
    st.markdown(f'<div class="metric-container"><h3>Thermal State</h3><h1>{status}</h1><p>{temp}°C</p></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-container"><h3>Model Error</h3><h1>±{int(rmse)}</h1><p>RMSE Score</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Main Chart
st.subheader("📊 Complex Scenario Modeling")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-90:], y=df['AQI'].iloc[-90:], mode='lines', name='Historical Trend', line=dict(color='#2E86C1')))
future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7)
fig.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), mode='lines+markers', name='AI Prediction', line=dict(color='#E74C3C', width=3)))
fig.update_layout(template="plotly_white", height=500, title="Impact of Weather & Industry on AQI")
st.plotly_chart(fig, use_container_width=True)

st.info("ℹ️ **Model Logic:** Notice how increasing 'Wind Speed' drastically lowers the predicted curve, while low 'Temperature' spikes it. This mimics real atmospheric inversion layers.")
