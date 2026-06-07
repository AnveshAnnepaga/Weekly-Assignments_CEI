# ⚡ Tesla Sales & Delivery Intelligence Platform
 
*(**Live Demo:** [https://teslasalesanddeliveryintelligenceplatform.streamlit.app/])*

## Project Overview
This project presents an enterprise-grade analytics dashboard that visualizes Tesla's historical sales, explores key vehicle metrics, and forecasts future deliveries using state-of-the-art machine learning models (Random Forest and Prophet). 

Designed to look and feel like an internal SaaS product used by Tesla's Operations and Supply Chain teams.

## ✨ Key Features
- **Delivery Prediction Engine:** A Random Forest model predicting vehicle delivery volumes based on pricing, battery capacity, range, and local charging infrastructure.
- **Scenario Simulator:** Interactive sliders allowing business leaders to adjust MSRP or production targets and instantly see the simulated impact on sales.
- **Future Forecasting:** Facebook Prophet integration that casts a 24-month horizon of expected future deliveries based on historical seasonality and trends.
- **AI Insights:** Dynamic insight cards explaining the mathematical "why" behind the predictions.
- **Premium UI/UX:** Built with a modern dark theme, glassmorphism CSS, animated KPI cards, and interactive Plotly charts.

## Structure
- `data/`: Contains raw dataset and processed forecast data.
- `notebooks/`: Jupyter notebooks used for data cleaning, EDA, and model prototyping.
- `model/`: Pickle files of the trained machine learning models (`random_forest_model.pkl` and `prophet_model.pkl`).
- `scripts/`: Training scripts to build the models without data leakage.
- `streamlit_app/`: The main application code and premium UI CSS assets.

## 🚀 How to Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the Models:**
*You must run this once to generate the `.pkl` files before starting the app!*
```bash
python scripts/train_models.py
```

3. **Launch the Intelligence Platform:**
```bash
streamlit run streamlit_app/app.py
```
