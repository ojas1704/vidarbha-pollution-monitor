# 🌱 Vidarbha Air Pulse (VAP)

**A Deep Learning-powered Air Quality Monitoring & Forecasting System.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vidarbha-pollution-monitor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%20%7C%20LSTM-orange)](https://www.tensorflow.org/)

## 📖 Overview
Vidarbha Air Pulse is an interactive web dashboard that monitors air pollution levels in major cities of the Vidarbha region (Nagpur, Amravati, Chandrapur, etc.). 

Beyond simple monitoring, it employs a **Long Short-Term Memory (LSTM)** Recurrent Neural Network (RNN) to predict future Air Quality Index (AQI) trends based on historical data. It allows users to simulate "what-if" scenarios by adjusting environmental stressors like traffic and construction to see real-time AI predictions.

## 🚀 Key Features

* **🧠 Deep Learning Engine:** Uses a custom-trained LSTM model to forecast AQI 7 days into the future.
* **🎛️ Scenario Simulator:** Interactive sliders let users adjust *Traffic Density* and *Construction Activity* to see how they impact the AI's forecast.
* **📊 Dynamic Visualization:** Real-time interactive charts and heatmaps using Plotly.
* **🌍 Regional Focus:** Specialized data simulation for the Vidarbha region's specific climatic conditions.

## 🛠️ Technical Architecture

The project is built using a modern Python stack:

* **Frontend:** [Streamlit](https://streamlit.io/) (for the interactive web interface).
* **Machine Learning:** [TensorFlow/Keras](https://www.tensorflow.org/) (LSTM model architecture).
* **Data Processing:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) (Time-series manipulation).
* **Visualization:** [Plotly](https://plotly.com/) (Interactive graphing).

### Why LSTM?
We chose Long Short-Term Memory (LSTM) networks because air quality data is a **Time Series** problem. Unlike standard regression models, LSTMs have a "memory" mechanism that captures long-term dependencies (e.g., pollution trends from the past 60 days) to accurately predict the next day's value.

## 💻 Installation & Local Run

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/vidarbha-pollution-monitor.git](https://github.com/your-username/vidarbha-pollution-monitor.git)
    cd vidarbha-pollution-monitor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

```text
├── app.py                # Main application code (UI + Model Training)
├── requirements.txt      # List of Python libraries required
├── README.md             # Project documentation
└── .gitignore            # Files to ignore (e.g., local venv)
