# 🛒 Supermarket Sales Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)

## 📌 Overview
The Supermarket Sales Forecasting Engine is a data-driven application designed to help retail managers optimize inventory and reduce perishable waste. By leveraging historical transaction data and machine learning, this tool accurately predicts future sales volumes for various vegetable categories, allowing for smarter supply chain decisions.

## ✨ Key Features
- **Machine Learning Predictions:** Utilizes a Random Forest Regressor to forecast sales demand based on seasonal trends, pricing, and product categories.
- **Historical Analysis:** Instantly filters past data to generate comprehensive monthly revenue and volume reports.
- **Automated Data Cleaning:** Built-in preprocessing that handles missing values and removes outliers using the Interquartile Range (IQR) method.
- **Interactive Dashboard:** A web-based GUI that allows users to seamlessly switch between viewing historical metrics and generating AI forecasts.
- **Visual Insights:** Dynamic, auto-generated bar charts and data tables for quick, intuitive decision-making.

## 🧠 Tech Stack
- **Core Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (RandomForestRegressor, LabelEncoder)
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit

## 📂 Project Structure
- `app.py`: The main application script containing the Streamlit UI and the Machine Learning pipeline.

## 📊 How the Model Works
1. **Data Ingestion:** The app reads the raw CSV files and strips formatting errors.
2. **Aggregation:** Transactions are grouped by Month and Item Code to establish historical baselines.
3. **Encoding:** Categorical text data (like vegetable names) is transformed into numerical values so the AI can process them.
4. **Training:** The Random Forest algorithm builds multiple decision trees to find non-linear relationships between month, price, and quantity sold.
5. **Output:** The model calculates the expected sales volume for a user-selected target month and displays the top 10 results visually.

### Dataset
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/yapwh1208/supermarket-sales-data)
