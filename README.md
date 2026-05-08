# Layoff & Leverage

**Analyzing how tech layoff announcements impact stock market performace**

This was my final project for IST 356 at Syracuse University. It follows
a full ETL (Extract, Transfer, Load) workflow and presents results through
an interactive Streamlit dashboard with a built-in predictive model.

## Project Overview

After a major tech company announces mass layoffs, does their stock tend to go up or down?
Through combining data from Layoffs.fyi with historical stock prices from Yahoo Finance, this
project explores that question. It also builds a machine learning model to predict stock movement 30-days
after a layoff announcement.

## Repository Structure

Layoffs & Leverage/
|
├── phase1_cleaning.py # ETL Pipeline saves data
├── transform.py # Reusable Transformation and Filter Functions
├── app.py # Streamlit Dashboard with Predictive Tool
├── predictive_model.py # Model Training, Evaluation, and Chart Generation
|
├── tests\
| └── test_transform.py # 28 Unit Tests for transform.py
|
├── layoffs.csv # Raw Dataset from Kaggly (layoffs.fyi)
├── layoffs_with_stocks.csv # Enriched Dataset Produced by phase1_cleaning.py
├── model_results.csv # Model Predictions Produced by predictive_model.py
|
├── visuals/ # Saved Chart Images
├── requirements.txt # Required Apps to Install
├── README.md # Project Documentation

## Data Sources

- [Layoffs.fyi Kaggle Dataset]
  (https://www.kaggle.com/datasets/swaptr/layoffs-2022)
- Yahoo Finance via yfinance

## Installation

**1. Clone the Repository:**
'''bash
git clone https://github.com/your-username/layoffs-and-leverage.git
cd layoffs-and-leverage
'''

**2. Install Dependencies:**
'''bash
pip install -r requirements.txt
'''

**3. Download the Dataset:**

- Download layoffs.csv from Kaggle
- Place it in the Root Project Folder

## Usage

**Step 1 - Run the ETL Pipeline**
(generates layoffs_with_stocks.csv):

'''bash
python3 phase1_cleaning.py
'''

**Step 2 - Launch the Streamlit Dashboard:**
'''bash
streamlit run app.py
'''

**Step 3 - (Optional) Run the Predictive Model:**
'''bash
python3 predictive_model.py
'''

**Step 4 - (Optional) Run the Unit Tests:**
'''bash
python3 -m pytest tests/test_transform.py -v
'''

# Features

**ETL Pipeline (phase1_cleaning.py)**

- Extracts Layoff Data from Kaggle CSV
- Maps Company Names to Stock Tickers
- Pulls Historical Stock Prices via yfinance
- Engineers Return Columns (30-day and Immediate) and Binary Target Variable
- Saves Enriched Dataset to layoffs_with_stocks.csv

**Streamlit Dashboard (app.py)**

- Sidebar Filters for Data Range, Industry, and Company
- 5 KPI Summary Cards
- 6 Interactive Charts Including Layoff Trends, Return Distributions, and
  Sector Comparisons
- Raw Data Table with CSV Download
- Stock Movement Predictor - Enter Layoff Details and get a Real-Time Predictor

**Predictive Models (predictive_model.py)**

- Logistic Regression (Baseline)
- Random Forest Classifier (Advanced)
- Cross-Validation, Confusion Matrices, Feature Importance Chart,
  and Model Comparison

**Unit Tests (tests/test_transform.py)**

- 28 Unit Tests Covering all 7 Functions in transform.py
- Run with python3 -m pytest tests/test_transform.py -v

## Running Tests

''' bash
python3 -m pytest tests/test_transform.py -v
'''

Expected Output:
28 Passed in 0.42s

## Tech Stack

| Tool         | Purpose                 |
| ------------ | ----------------------- |
| Python 3.11  | Core Language           |
| Pandas       | Data Transformation     |
| yfinance     | Stock Price Extraction  |
| Streamlit    | Interactive Dashboard   |
| Matplotlib   | Data Visualization      |
| scikit-learn | Machine Learning Models |
| pytest       | Unit Testing            |

## Author

Sachin Sastry

## Disclaimer

This Project is for Eductional Purposes Only!

Stock predictions made by this model should not be used as financial advice.
