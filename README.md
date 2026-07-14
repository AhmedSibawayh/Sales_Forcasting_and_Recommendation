# Walmart Weekly Sales Forecasting & Business Recommendation System

An end-to-end Machine Learning project that predicts Walmart weekly sales using historical data and provides business recommendations to support inventory planning, staffing, and promotional decisions.

The project also includes MLflow experiment tracking and a FastAPI deployment for real-time predictions.

---

## Project Overview

Retail companies need accurate demand forecasting to optimize inventory, reduce waste, improve staffing, and maximize revenue.

This project uses the **Walmart Recruiting Store Sales Forecasting** dataset to build a complete machine learning pipeline:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple Machine Learning Models
- MLflow Experiment Tracking
- Business Recommendation Engine
- FastAPI Deployment

---

## Dataset

Dataset:
**Walmart Recruiting - Store Sales Forecasting**

https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

The dataset contains:

- Store ID
- Department ID
- Weekly Sales
- Date
- Holiday Indicator
- Temperature
- Fuel Price
- CPI
- Unemployment
- Markdown Features
- Store Type
- Store Size

---

## Project Workflow

```text
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
MLflow Experiment Tracking
    │
    ▼
Business Recommendation Engine
    │
    ▼
FastAPI Deployment
```

---

## Exploratory Data Analysis

The project includes several visualizations, including:

- Weekly Sales Distribution
- Sales Trend Over Time
- Holiday vs Non-Holiday Sales
- Monthly Sales Analysis
- Department Sales Comparison
- Store Type Comparison
- Store Size Analysis
- Correlation Heatmap

---

## Feature Engineering

Several time-series features were created to improve forecasting accuracy.

### Calendar Features

- Year
- Month
- Day
- Day of Week
- Week of Year

### Lag Features

- Sales_Lag_1
- Sales_Lag_2
- Sales_Lag_3

### Rolling Statistics

- Sales_MA_4
- Sales_MA_8

### Categorical Encoding

- One-Hot Encoding of Store Type

---

## Machine Learning Models

The following models were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

The best-performing model was selected based on evaluation metrics.

---

## Evaluation Metrics

The models were evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## Business Recommendation System

The Walmart dataset does not include customer-level purchase history.

Instead of collaborative filtering, this project implements a business recommendation engine that generates operational recommendations based on forecasted demand.

Example recommendations include:

### High Demand

- Increase inventory
- Schedule additional staff
- Increase promotional budget

### Medium Demand

- Maintain current inventory
- Continue existing staffing levels

### Low Demand

- Reduce inventory
- Delay replenishment
- Optimize staffing

---

## MLflow Integration

The project uses MLflow to track experiments, including:

- Hyperparameters
- Evaluation Metrics
- Trained Models
- Artifacts
- Feature Importance
- Champion Model Selection

---

## FastAPI Deployment

The trained model is deployed using FastAPI.

Available endpoints:

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | / | API Status |
| POST | /predict | Predict Weekly Sales |
| POST | /predict-csv | Batch Prediction using CSV |

Run the API locally:

```bash
uvicorn main:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- MLflow
- FastAPI
- Matplotlib
- Seaborn

---

## Repository Structure

```
├── API/
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│
├── notebooks/
│
├── images/
│
├── data/
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

- TimeSeriesSplit Cross Validation
- LSTM Forecasting
- Facebook Prophet Comparison
- Docker Deployment
- Cloud Deployment
- Automated Retraining Pipeline
- Real Customer Recommendation System

---

## Author

Ahmed

Chemical Engineering Graduate | Machine Learning & Data Science

LinkedIn:
(Add your LinkedIn profile here)
