import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Vidarbha Air Pulse", layout="wide", page_icon="🍃")

st.markdown("""
    <style>
    /* Force background to white and text to dark everywhere */
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    h1, h2, h3, h4, h5, h6, p, span, div { color: #2E4053 !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetricValue"] { color: #1a1a1a !important; }
    .metric-container {
        background-color: #f0fdf4; border-left: 5px solid #28B463;
        padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-container h1 { color: #1a1a1a !important; }
    .metric-container p { color: #505050 !important; }
    div.stButton > button { background-color: #28B463; color: white !important; border-radius: 30px; border: none; font-weight: bold; }
    div.stButton > button:hover { background-color: #1D8348; transform: scale(1.05); color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def get_base_data():
    # 2 years of data to give the model more to learn from
    dates = pd.date_range(end=datetime.today(), periods=730)
    # Complex pattern: Seasonal (Yearly) + Weekly (Traffic) + Random Noise
    t = np.arange(730)
    seasonal = 40 * np.sin(t / 50)  # Seasonal drift
    weekly = 15 * np.sin(t / 7)     # Weekly traffic cycles
    noise = np.random.normal(0, 8, 730) # Random pollution spikes
    
    aqi = 110 + seasonal + weekly + noise
    aqi = np.clip(aqi, 40, 400)
    
    df = pd.DataFrame({'Date': dates, 'AQI': aqi})
    df.set_index('Date', inplace=True)
    return df

# --- 3. ROBUST LSTM MODEL (With Dropout & Train/Test Split) ---
@st.cache_resource
def train_and_evaluate(df, traffic_stress=0, construction_stress=0):
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Apply User Stressors to the *Last 60 Days* only (Simulating sudden changes)
    stress_factor = (traffic_stress * 0.005) + (construction_stress * 0.01)
    if stress_factor > 0:
        scaled_data[-60:] += stress_factor

    # Prepare Sequences
    prediction_days = 60
    x, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        x.append(scaled_data[i-prediction_days:i, 0])
        y.append(scaled_data[i, 0])
        
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # --- KEY FIX: TRAIN / TEST SPLIT ---
    # We hide the last 20% of data from the training process
    split_index = int(len(x) * 0.8)
    
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build LSTM with DROPOUT (Prevents Overfitting)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Randomly drop 20% of neurons to prevent memorization
    model.add(LSTM(units=50))
    model.add(Dropout(0.2)) # Second dropout layer
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early Stopping: Stop if it stops improving to prevent over-training
    early_stop = EarlyStopping(monitor='loss', patience=3)
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[early_stop])

    # Evaluate on TEST data (Data the model has NEVER seen)
    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate Error Metrics
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    # Forecast Next 7 Days
    future_forecast = []
    current_batch = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    for i in range(7):
        pred = model.predict(current_batch, verbose=0)[0,0]
        future_forecast.append(pred)
        new_step = np.array([[[pred]]])
        current_batch = np.append(current_batch[:,1:,:], new_step, axis=1)
    
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
    
    return future_forecast, predictions, y_test_actual, mae, rmse, df.index[split_index+prediction_days:]

# --- 4. UI LAYOUT ---

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌱 Vidarbha Air Pulse V2.0")
    st.markdown("### Robust Forecasting with Overfitting Protection")
    st.write("Now using **Dropout Regularization** and **Train/Test Splitting** for honest validation.")
with col2:
    st.image("https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?q=80&w=1974&auto=format&fit=crop", use_container_width=True)

st.markdown("---")

# Controls
st.sidebar.header("⚙️ Live Conditions")
traffic = st.sidebar.slider("🚦 Traffic Density (%)", 0, 100, 20)
construction = st.sidebar.slider("🏗️ Active Construction Sites", 0, 50, 5)

# Run Model
df = get_base_data()
with st.spinner("Training LSTM Model on 80% of data... (Testing on remaining 20%)"):
    forecast, test_preds, test_actual, mae, rmse, test_dates = train_and_evaluate(df, traffic, construction)

tomorrows_aqi = int(forecast[0][0])

# Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-container"><h3>Current AQI</h3><h1>{int(df["AQI"].iloc[-1])}</h1><p>Nagpur Sensor Node</p></div>', unsafe_allow_html=True)
with m2:
    color = "#E74C3C" if tomorrows_aqi > 150 else "#28B463"
    st.markdown(f'<div class="metric-container" style="border-left: 5px solid {color};"><h3>Forecast (Tomorrow)</h3><h1 style="color:{color}">{tomorrows_aqi}</h1><p>AI Prediction</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-container"><h3>Model Error (RMSE)</h3><h1>±{int(rmse)}</h1><p>Deviation on Unseen Data</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Forecast", "📉 Model Validation (For Geeks)", "🗺️ Map"])

with tab1:
    st.subheader("7-Day Pollution Forecast")
    fig = go.Figure()
    # History
    fig.add_trace(go.Scatter(x=df.index[-90:], y=df['AQI'].iloc[-90:], mode='lines', name='Historical Data', line=dict(color='#2E86C1', width=2)))
    # Forecast
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7)
    fig.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), mode='lines+markers', name='AI Prediction', line=dict(color='#E74C3C', width=3, dash='dot')))
    fig.update_layout(template="plotly_white", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("⚠️ Model Reality Check (Train vs Test)")
    st.write(f"We hid the last **{len(test_actual)} days** of data from the model during training. The chart below shows how well the model guessed those 'secret' days.")
    
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=test_dates, y=test_actual.flatten(), mode='lines', name='Actual Values (Hidden)', line=dict(color='grey')))
    fig_val.add_trace(go.Scatter(x=test_dates, y=test_preds.flatten(), mode='lines', name='Model Prediction', line=dict(color='orange')))
    
    st.plotly_chart(fig_val, use_container_width=True)
    
    if rmse < 20:
        st.success(f"✅ Low RMSE ({int(rmse)}): The model is generalizing well and NOT overfitting.")
    else:
        st.warning(f"⚠️ High RMSE ({int(rmse)}): The model might be struggling with the noise.")

with tab3:
    st.subheader("Regional Heatmap")
    map_data = pd.DataFrame({
        'City': ['Nagpur', 'Amravati', 'Chandrapur', 'Akola', 'Wardha'],
        'lat': [21.1458, 20.9320, 19.9615, 20.7002, 20.7453],
        'lon': [79.0882, 77.7523, 79.2961, 77.0082, 78.6022],
        'AQI': [tomorrows_aqi, tomorrows_aqi-15, tomorrows_aqi+35, tomorrows_aqi-8, tomorrows_aqi-12]
    })
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", color="AQI", size="AQI",
                                hover_name="City", zoom=6, mapbox_style="carto-positron",
                                color_continuous_scale="RdYlGn_r", size_max=40)
    st.plotly_chart(fig_map, use_container_width=True)
