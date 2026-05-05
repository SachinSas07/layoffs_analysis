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

    
class TestMapTickers: # Tests for map_tickers function

    def test_adds_ticker_column(self, clean_df):
        """The map_tickers function should add a 'ticker' column to the DataFrame"""
        result = map_tickers(clean_df)
        assert "ticker" in result.columns, "Expected 'ticker' column to be added by map_tickers"

    def test_drops_unknown_companies(self, clean_df):
        """Companies not in the ticker map should be dropped"""
        result = map_tickers(clean_df)
        assert "PrivateCo" not in result["company"].values, \
            "Expected unknown companies to be dropped by map_tickers"
        
    def test_known_companies_mapped_correctly(self, clean_df):
        """Known companies should be mapped to their correct tickers"""
        result = map_tickers(clean_df)
        meta_row = result[result["company"] == "Meta"]
        assert not meta_row.empty, "Expected Meta to be present in the mapped DataFrame"
        assert meta_row["ticker"].iloc[0] == "META", \
            "Expected Meta to be mapped to ticker 'META'"
    
    def test_no_null_tickers_in_output(self, clean_df):
        """After mapping tickers, there should be no null values in the 'ticker' column"""
        result = map_tickers(clean_df)
        assert result["ticker"].isnull().sum() == 0, \
            "Expected no null values in 'ticker' column after mapping"

class TestCalculateReturns:

    def test_return_immediate_calculated_correctly(self):
        """The return_immediate column should be calculated as (price_day0 - price_before) / price_before * 100"""
        df = pd.DataFrame({
            "price_before" : [100.0],
            "price_day0"   : [110.0],
            "price_day30"  : [120.0]
        })
        result = calculate_returns(df)
        assert abs(result["return_30day"].iloc[0] - 20.0) < 0.01, \
            "Expected 30-day return of 20%"
        
    def test_return_30day_calculated_correctly(self):
        """The return_30day column should be calculated as (price_day30 - price_before) / price_before * 100"""
        df = pd.DataFrame({
            "price_before" : [100.0],
            "price_day0"   : [110.0],
            "price_day30"  : [120.0]
        })
        result = calculate_returns(df)
        assert abs(result["return_30day"].iloc[0] - 20.0) < 0.01, \
            "Expected 30-day return of 20%"
        
    def test_target_is_1_when_stock_up(self):
        """The target column should be 1 when 30-day return is positive"""
        df = pd.DataFrame({
            "price_before" : [100.0],
            "price_day0"   : [105.0],
            "price_day30"  : [115.0]
        })
        result = calculate_returns(df)
        assert result["target"].iloc[0] == 1, \
            "Expected target to be 1 when stock price goes up after layoff event"
        
    def test_target_is_0_when_stock_down(self):
        """The target column should be 0 when 30-day return is negative"""
        df = pd.DataFrame({
            "price_before" : [100.0],
            "price_day0"   : [97.0],
            "price_day30"  : [95.0]
        })
        result = calculate_returns(df)
        assert result["target"].iloc[0] == 0, \
            "Expected target to be 0 when stock price goes down after layoff event"
        
    def test_adds_all_3_columns(self):
        """The calculate_returns function should add 'return_immediate', 'return_30day', and 'target' columns to the DataFrame"""
        df = pd.DataFrame({
            "price_before" : [100.0],
            "price_day0"   : [105.0],
            "price_day30"  : [110.0]
        })
        result = calculate_returns(df)
        for col in ["return_immediate", "return_30day", "target"]:
            assert col in result.columns, f"Expected '{col}' column to be added by calculate_returns"

# Filter by Date
class TestFilterByDate:

    def test_filters_correctly(self, enriched_df):
        """The filter_by_date function should correctly filter so only rows within the data range remain"""
        result = filter_by_date(enriched_df, "2023-01-01", "2023-02-01")
        assert len(result) == 1, "Expected only 1 row to remain after filtering by date"
        assert result.iloc[0]["company"] == "Meta", "Expected the remaining row to be for Meta"

    def test_empty_result_outside_range(self, enriched_df):
        """No Rows should remain if the date range does not include any of the rows in the DataFrame"""
        result = filter_by_date(enriched_df, "2020-01-01", "2020-12-31")
        assert result.empty, "Expected an empty DataFrame when filtering by a date range with no matching rows"

    def test_returns_dataframe(self, enriched_df):
        """The output of filter_by_date should be a pandas DataFrame"""
        result = filter_by_date(enriched_df, "2023-01-01", "2023-12-31")
        assert isinstance(result, pd.DataFrame), "Expected the output of filter_by_date to be a DataFrame"

# Filter by Industry
class TestFilterByIndustry:

    def test_filters_to_selected_industry(self, enriched_df):
        """The filter_by_industry function should return only rows where the industry matches the selected industry"""
        result = filter_by_industry(enriched_df, ["Technology"])
        assert all(result["industry"] == "Technology"), \
            "Expected all rows to have industry 'Technology' after filtering"
    
    def test_empty_list_returns_all_rows(self, enriched_df):
        """If an empty list is passed to filter_by_industry, all rows should be returned"""
        result = filter_by_industry(enriched_df, [])
        assert len(result) == len(enriched_df), \
            "Expected all rows to be returned when filtering with an empty industry list with no filters"
        
    def test_multiple_industries_filtered_correctly(self, enriched_df):
        """The filter_by_industry function should correctly filter to multiple selected industries"""
        result = filter_by_industry(enriched_df, ["Consumer", "Technology"])
        assert set(result["industry"].unique()) == {"Consumer", "Technology"}, \
            "Expected only 'Consumer' and 'Technology' industries to be present after filtering"
        
# Filter by Company
class TestFilterByCompany:

    def test_filters_to_selected_company(self, enriched_df):
        """The filter_by_company function should return only rows where the company matches the selected company"""
        result = filter_by_company(enriched_df, ["Meta"])
        assert all(result["company"] == "Meta"), \
            "Expected only rows with company 'Meta' to be present after filtering"
    
    def test_empty_list_returns_all_rows(self, enriched_df):
        """If an empty list is passed to filter_by_company, all rows should be returned"""
        result = filter_by_company(enriched_df, [])
        assert len(result) == len(enriched_df), \
            "Expected all rows to be returned when filtering with an empty company list with no filters"
        
    def test_multiple_companies_filtered_correctly(self, enriched_df):
        """The filter_by_company function should correctly filter to multiple selected companies"""
        result = filter_by_company(enriched_df, ["Meta", "Google"])
        assert set(result["company"].unique()) == {"Meta", "Google"}, \
            "Expected only 'Meta' and 'Google' companies to be present after filtering"
        
# Get summary stats
class TestGetSummaryStats:

    def test_returns_dict(self, enriched_df):
        """The output should be a dictionary"""
        result = get_summary_stats(enriched_df)
        assert isinstance(result, dict), "Expected the output of get_summary_stats to be a dictionary"
        
    def test_empty_df_returns_empty_dict(self):
        """If an empty DataFrame is passed, the output should be an empty dictionary"""
        result = get_summary_stats(pd.DataFrame())
        assert result == {}, "Expected an empty dictionary when passing an empty DataFrame to get_summary_stats"

    def test_total_events_correct(self, enriched_df):
        """The total_events key should correctly count the number of rows in the DataFrame"""
        result = get_summary_stats(enriched_df)
        assert result["total_events"] == len(enriched_df)
    
    def test_total_laid_off_correct(self, enriched_df):
        """The total_laid_off key should correctly sum the total_laid_off column in the DataFrame"""
        result = get_summary_stats(enriched_df)
        assert result["total_laid_off"] == int(enriched_df["total_laid_off"].sum())
        
    def test_pct_stock_up_is_percentage(self, enriched_df):
        """The pct_stock_up key should be a percentage between 0 and 100"""
        result = get_summary_stats(enriched_df)
        assert 0 <= result["pct_stock_up"] <= 100, \
            "Expected pct_stock_up expected to be a a value between 0 and 100"
        
    def test_contains_all_keys(self, enriched_df):
        """The output dictionary should contain the keys"""
        expected_keys = [
            "total_events", "total_laid_off", "avg_immediate_return",
            "avg_30day_return", "pct_stock_up", "largest_layoff_co",
            "largest_layoff_n", "best_return_co", "best_return_val",
            "worst_return_co", "worst_return_val"
        ]
        result = get_summary_stats(enriched_df)
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' to be present in the summary stats dictionary"