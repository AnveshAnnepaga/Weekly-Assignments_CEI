# Tesla Sales & Delivery Forecasting

## Project Overview
This project presents an enterprise-grade analytics dashboard that visualizes Tesla's historical sales, explores key vehicle metrics, and forecasts future deliveries using state-of-the-art machine learning models (Random Forest and Prophet).

## Dashboard Architecture
Built with Python and **Streamlit**, this dashboard focuses on providing a clean, SaaS-like experience with premium glassmorphism aesthetics and modern interactive Plotly charts. 

## Structure
- `data/`: Contains raw dataset and processed forecast data.
- `notebooks/`: Jupyter notebooks used for data cleaning and model training.
- `models/`: Pickle files of the trained models.
- `reports/`: Documentation and business insights.
- `streamlit_app/`: The main application code and UI assets.

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Navigate to the `streamlit_app` folder and run the app:
```bash
cd streamlit_app
streamlit run app.py
```
