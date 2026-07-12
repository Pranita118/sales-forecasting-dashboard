# Sales Forecasting & Demand Intelligence System

An end-to-end sales forecasting system built on the Superstore Sales dataset — covering time series analysis, three forecasting models, anomaly detection, product demand segmentation, and a deployed interactive dashboard.

**Live Dashboard:** [https://sales-forecasting-dashboard-kbrmqckyrrwdjwrkvhjkc6.streamlit.app/](https://sales-forecasting-dashboard-kbrmqckyrrwdjwrkvhjkc6.streamlit.app/)

## Project Overview

This project answers the core retail question: *how much of each product will we sell next month, and will we have enough stock to meet that demand?* It walks through the full pipeline a real data science team would build — from raw sales data to a forecast a supply chain manager could act on Monday morning.

## What's Included

| File | Description |
|---|---|
| `analysis.ipynb` | Full Jupyter notebook covering all 8 project tasks, with code and written interpretation |
| `app.py` | Streamlit dashboard — 4 pages (Sales Overview, Forecast Explorer, Anomaly Report, Demand Segments) |
| `requirements.txt` | Python dependencies for running the app |
| `train.csv` | Superstore Sales dataset |
| `vgsales.csv` | Supplementary dataset used for multi-source merging practice |
| `summary.docx` | 2-page executive business report for non-technical stakeholders |
| `charts/` | All generated chart images (PNG) |
| `.streamlit/config.toml` | Light theme configuration for the dashboard |

## Key Findings

- **Top revenue category:** Technology ($827,456 over 4 years)
- **Most consistent regional growth:** East region
- **Seasonality:** Sales spike reliably every November and December
- **Best forecasting model:** SARIMA (lowest MAE, RMSE, and MAPE of the three models tested — beating Prophet and XGBoost)
- **3-month forecast:** $73,694 → $95,397 → $107,211
- **Anomaly detection:** Isolation Forest and Z-score methods agreed on 92.8% of weekly data points
- **Product segmentation:** 4 demand clusters identified — High Volume/Stable, High Value/Volatile, Low Volume/Stable, and Niche/Rapidly Growing — each with a distinct recommended stocking strategy

## Tech Stack

- **Data & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels, Prophet, pmdarima
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit, Streamlit Community Cloud

## Running Locally

```bash
git clone https://github.com/YOUR-USERNAME/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Notebook

Open `analysis.ipynb` in Jupyter or Google Colab to see the full analysis, including:
1. Data loading, cleaning, and exploratory analysis
2. Time series decomposition and stationarity testing
3. Three forecasting models compared head-to-head (SARIMA, Prophet, XGBoost)
4. Category- and region-level forecasts
5. Anomaly detection using two independent methods
6. Product demand segmentation via K-Means clustering
7. Dashboard deployment code

## Known Limitations

This system is trained entirely on historical patterns and cannot anticipate unprecedented events — a new competitor, a supply chain disruption, or a sharp economic shift. Forecasts should be refreshed monthly and treated as decision support, not a guarantee. One specific caution: the category-level forecast for Furniture showed an unusually large swing that is likely an artifact of a small, noisy sample rather than a genuine trend.

---

*Built as part of a Data Science internship project — Week 3 & 4: End-to-End Sales Forecasting & Demand Intelligence System.*
