import pytest 
import pandas as pd
import sys
import os

# Add the parent directory to the system path to allow importing transform.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transform import (
    clean_dataframe,
    map_tickers,
    calculate_returns,
    filter_by_date,
    filter_by_industry,
    filter_by_company,
    get_summary_stats
)

# Fixture to create a sample DataFrame for testing

@pytest.fixture
def raw_df():
    """A small sample of raw layoff data mimicking the Kaggle format"""
    return pd.DataFrame({
        "company"           : ["Meta", "Google", "PrivateCo", "Amazon", "Meta"],
        "location"          : ["SF Bay Area", "SF Bay Area", "NYC", "Seattle", "SF Bay Area"],
        "total_laid_off"    : [11000, 12000, 500, None, 3000],
        "date"              : ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "percentage_laid_off" : [0.13, 0.06, None, None, 0.25],
        "industry"          : ["Consumer", "Technology", "Finance", "Consumer", "Consumer"],
        "source"            : [None, None, None, None, None],
        "stage"             : ["POST-IPO", "POST-IPO", "Series B", "POST-IPO", "POST-IPO"],
        "funds_raised"       : [None, None, 50, None, None],
        "country"           : ["United States", "United States", "United States", "United States", "United States"],
        "date_added"        : ["2023-01-02", "2023-02-02", "2023-03-02", "2023-04-02", "2023-05-02"]
    })

@pytest.fixture
def clean_df(raw_df):
    """A cleaned version of the raw DataFrame after applying clean_dataframe()"""
    return clean_dataframe(raw_df)

@pytest.fixture
def enriched_df():
    """A small dataframe that has price and return columns"""
    return pd.DataFrame({
        "company"        : ["Meta", "Google", "Amazon"],
        "ticker"         : ["META", "GOOGL", "AMZN"],
        "industry"       : ["Consumer", "Technology", "Consumer"],
        "date"           : pd.to_datetime(["2023-01-01", "2023-03-15","2023-06-01"]),
        "total_laid_off" : [11000, 12000, 5000],
        "price_before"   : [100, 200, 150],
        "price_day0"     : [105, 196, 103],
        "price_day30"    : [100, 190, 140],
        "return_immediate": [5.0, -2.0, 2],
        "return_30day"   : [10.0, -5.0, -6.67],
        "target"         : [1, 0, 0]
    })

# Tests for clean_dataframe function

class TestCleanDataframe:

    def test_date_converted_to_datetime(self, clean_df):
        """The date column should be converted to datetime format"""
        result = clean_dataframe(clean_df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"]), \
        "Expected 'date' column to be datetime type after cleaning"

    def test_drops_rows_with_null_total_laid_off(self, clean_df):
        """Rows with null values in 'total_laid_off' should be dropped"""
        result = clean_dataframe(clean_df)
        assert result["total_laid_off"].isnull().sum() == 0, \
            "Expected no null values in 'total_laid_off' after cleaning"
        
    def test_filters_to_us_only(self):
        """Only rows where country == 'United States' should be retained"""
        df = pd.DataFrame({
            "company"          : ["Meta", "Acko"],
            "total_laid_off"   : [1000, 200],
            "date"             : ["2023-01-01", "2023-02-01"],
            "country"          : ["United States", "India"]
        })
        result = clean_dataframe(clean_df)
        assert all(result["country"] == "United States"), \
            "Expected all rows to have country as 'United States' after cleaning"
        
    def test_output_is_dataframe(self, raw_df):
        """The output of clean_dataframe should be a pandas DataFrame"""
        result = clean_dataframe(raw_df)
        assert isinstance(result, pd.DataFrame)