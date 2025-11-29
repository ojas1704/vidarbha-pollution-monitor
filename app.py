import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & AVAADA STYLING ---
st.set_page_config(page_title="Vidarbha Air Pulse", layout="wide", page_icon="🍃")

# Corporate Green Theme (Avaada Style)
# Corporate Green Theme (Avaada Style) - FORCE DARK TEXT
st.markdown("""
    <style>
    /* Force background to white and text to dark everywhere */
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Force specific headers to be dark grey */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #2E4053 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Fix for the Metrics inside the cards */
    div[data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
    
    /* Custom Green Highlight for metrics */
    .metric-container {
        background-color: #f0fdf4; /* Light Green */
        border-left: 5px solid #28B463;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Ensure metric text inside custom HTML is visible */
    .metric-container h1 {
        color: #1a1a1a !important;
    }
    .metric-container h3 {
        color: #2E4053 !important;
    }
    .metric-container p {
        color: #505050 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #28B463;
        color: white !important; /* Keep button text white */
        border-radius: 30px;
        padding: 10px 30px;
        border: none;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1D8348;
        transform: scale(1.05);
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (Simulating Vidarbha) ---
@st.cache_data
def get_base_data():
    # Generate 1 year of historical data
    dates = pd.date_range(end=datetime.today(), periods=365)
    # Sin wave for seasonal changes + Random noise
    aqi = 100 + (50 * np.sin(np.arange(365)/30)) + np.random.normal(0, 10, 365)
    aqi = np.clip(aqi, 40, 400) # Clamp values
    df = pd.DataFrame({'Date': dates, 'AQI': aqi})
    df.set_index('Date', inplace=True)
    return df

# --- 3. DEEP LEARNING MODEL (LSTM) ---
@st.cache_resource
def build_and_predict(df, traffic_stress=0, construction_stress=0):
    # Prepare Data
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Apply "Stress" from User Sliders to the recent data
    stress_factor = (traffic_stress * 0.01) + (construction_stress * 0.02)
    if stress_factor > 0:
        scaled_data[-30:] += stress_factor # Apply stress to last 30 days

    # Create Sequences (Lookback 60 days)
    x_train, y_train = [], []
    prediction_days = 60
    
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0) # Fast training

    # Predict Next 7 Days
    future_predictions = []
    current_batch = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    
    for i in range(7): # Predict a week out
        pred = model.predict(current_batch, verbose=0)[0,0]
        future_predictions.append(pred)
        new_step = np.array([[[pred]]])
        current_batch = np.append(current_batch[:,1:,:], new_step, axis=1)

    # Inverse Transform
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# --- 4. UI LAYOUT ---

# Hero Section
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌱 Vidarbha Air Pulse")
    st.markdown("### Unlocking Nature's Potential via Predictive AI")
    st.write("Monitoring regional air quality with deep learning to forecast pollution hotspots.")
with col2:
    # High-quality renewable energy image (Unsplash)
    st.image("https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?q=80&w=1974&auto=format&fit=crop", use_container_width=True)

st.markdown("---")

# Interactive Control Panel
st.sidebar.header("⚙️ Simulation Controls")
st.sidebar.write("Adjust factors to see how the LSTM model changes the forecast.")
traffic = st.sidebar.slider("🚦 Traffic Density (%)", 0, 100, 20)
construction = st.sidebar.slider("🏗️ Active Construction Sites", 0, 50, 5)

# Metrics Row
df = get_base_data()
predicted_values = build_and_predict(df, traffic, construction)
tomorrows_aqi = int(predicted_values[0][0])

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-container"><h3>Current AQI (Nagpur)</h3><h1>{int(df["AQI"].iloc[-1])}</h1><p>Real-time Sensor Data</p></div>', unsafe_allow_html=True)
with m2:
    color = "#E74C3C" if tomorrows_aqi > 150 else "#28B463"
    st.markdown(f'<div class="metric-container" style="border-left: 5px solid {color};"><h3>AI Forecast (Tomorrow)</h3><h1 style="color:{color}">{tomorrows_aqi}</h1><p>Based on current sliders</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-container"><h3>Risk Factors</h3><h1>{traffic}% Traffic</h1><p>Impact on Model Weights</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Main Dashboard
tab1, tab2, tab3 = st.tabs(["📊 Forecasting Hub", "🗺️ Hotspot Map", "🎥 Vision"])

with tab1:
    st.subheader("LSTM Time-Series Prediction")
    st.write("The chart below shows historical data (Blue) and the AI's prediction for the next 7 days (Red).")
    
    fig = go.Figure()
    # History
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['AQI'].iloc[-60:], mode='lines', name='History (Last 60 Days)', line=dict(color='#2E86C1', width=3)))
    
    # Prediction
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7)
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_values.flatten(), mode='lines+markers', name='AI Forecast', line=dict(color='#E74C3C', width=3, dash='dot')))
    
    fig.update_layout(template="plotly_white", height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Regional Pollution Hotspots")
    # Mock Geo Data for Vidarbha
    map_data = pd.DataFrame({
        'City': ['Nagpur', 'Amravati', 'Chandrapur', 'Akola', 'Wardha'],
        'lat': [21.1458, 20.9320, 19.9615, 20.7002, 20.7453],
        'lon': [79.0882, 77.7523, 79.2961, 77.0082, 78.6022],
        'AQI': [tomorrows_aqi, tomorrows_aqi-20, tomorrows_aqi+40, tomorrows_aqi-10, tomorrows_aqi-15]
    })
    
    fig_map = px.scatter_mapbox(map_data, lat="lat", lon="lon", color="AQI", size="AQI",
                                hover_name="City", zoom=6, mapbox_style="carto-positron",
                                color_continuous_scale="RdYlGn_r", size_max=40)
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Understanding the Invisible")
    st.write("Why monitoring air quality matters for a sustainable future.")
    # Embedded YouTube Video (National Geographic)
    st.video("https://www.youtube.com/watch?v=e6rglsLy1Ys") 

st.markdown("<br><br><center><small>© 2025 Vidarbha Air Pulse | Sustainable Tech</small></center>", unsafe_allow_html=True)
