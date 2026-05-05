import pandas as pd
import yfinance as yf
from datetime import timedelta
import time
'''
The purpose of this script is to load and clean the raw data from kaggle. 
It will clean and format the data, map the company names to their corresponding stock tickers, and save the cleaned data to a new CSV file for further analysis. 
It will pull historical stock price data for each company using the yfinance library. 
It will calculate stock returns around each layoff announcement. 
It will then save the final merged dataset for use in stage2. 
'''

# Load + Inspect 
# Load CSV file into a pandas DataFrame 
df = pd.read_csv('layoffs.csv')

# Print basic info to allow us to understand the dataset at a glance
print("Shape", df.shape) # Shape of the dataset (rows, columns)
print("\nColumn types:\n", df.dtypes) # Data types of each column
print("\nMissing values:\n", df.isnull().sum()) # Count of missing values in each column
print("\nSample Rows:")
print(df.head()) # Display the first few rows of the dataset

# Clean + Format
# Convert 'Date' column to datetime format
# error='coerce' will convert any non-date values to NaT (Null) instead of throwing an error
df["date"] = pd.to_datetime(df["date"], errors='coerce')

# Remove rows where we're missing company name, date, or layoff count since these are critical for our analysis
df = df.dropna(subset=["company", "date", "total_laid_off"])

# Filter to US based companies only. Makes easier since we are only dealing with US based companies and stock tickers
df = df[df["country"] == "United States"].copy()

# Reset index so row numbers are sequential after dropping rows
df = df.reset_index(drop=True)

print(f"\nRows After Cleaning: {len(df)}")

# Ticker Mapping
# Create a mapping of company names to their stock tickers because yfinance requires tickers to pull stock data
# For simplicity, we'll create a manual mapping for the most common companies in our dataset.
ticker_map = {
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Meta": "META",
    "Apple": "AAPL",
    "IBM": "IBM",
    "Salesforce": "CRM",
    "Twitter": "TWTR",
    "Uber": "UBER",
    "Netflix": "NFLX",
    "Salesforce": "CRM",
    "Twitter": "TWTR",
    "Lyft": "LYFT",
    "Snap": "SNAP",
    "Spotify": "SPOT",
    "Zillow": "Z",
    "Robinhood": "HOOD",
    "Coinbase": "COIN",
    "Peloton": "PTON",
    "Opendoor": "OPEN",
    "Intel": "INTC",
    "Cisco": "CSCO",
    "Oracle": "ORCL",
    "PayPal": "PYPL",
    "Shopify": "SHOP",
    "Zoom": "ZM",
    "DocuSign": "DOCU",
    "Twilio": "TWLO",
    "Rivian": "RIVN",
    "Lucid": "LCID",
    "Unity": "U",
    "Roblox": "RBLX",
    "DoorDash": "DASH",
    "Instacart": "CART",
    "Airbnb": "ABNB",
    "Expedia": "EXPE",
    "eBay": "EBAY",
    "Dell": "DELL",
    "HP": "HPQ",
    "Wayfair": "W",
    "Vimeo": "VMEO",
    "Warner Bros. Discovery": "WBD",
    "Disney": "DIS",
    "Comcast": "CMCSA",
    # Add more mappings as needed based on the companies in your dataset
}
# Add a new column 'ticker' to the DataFrame by mapping the 'company' column using our ticker_map
# Companies not in the ticker_map will get a NaN value for their ticker
df["ticker"] = df["company"].map(ticker_map)

# Keep only rows where we successfully mapped a ticker (i.e., where 'ticker' is not null).
# Private or those not in our map will be dropped here
df_public = df.dropna(subset=["ticker"]).copy().reset_index(drop=True)

print(f"Public Companies Matched: {len(df_public)}")
print(df_public[["company", "ticker", "date", "total_laid_off", "percentage_laid_off"]].head(10))

# Pulling Stock Data via Yahoo Finance (yfinance)
def get_price_window(ticker, event_date, before=1, after=30):
    """
    Fetches stock closing prices for a given ticker around a layoff announcement date.
    Parameters:
        ticker      : Stock ticker symbol (e.g., 'AAPL')
        event_date  : Date of the layoff announcement (datetime object)
        before      : how many days to look back before the event date
        after       : how many days after the event date to look forward
        
    Returns:
        price_before : closing price on the day before the announcement
        price_day0   : closing price on just after the announcement
        price_day30  : closing price 30 days after the announcement
    """
    # Add a buffer of extra days. This will account for weekends and holidays.
    start = event_date - timedelta(days=before + 5)
    end = event_date + timedelta(days=after + 5)

    try: 
    # Use yfinance to download historical stock data for the specified ticker and date range
        hist = yf.download(ticker, start=start, end=end, progress=False)

    # If no data is returned, skip this row
        if hist.empty:
            return None, None, None
    
        hist.index = pd.to_datetime(hist.index) # Ensure the index is in datetime format
    
        # Extract just closing price column 
        close = hist["Close"]

        # Get the closing price on the day before the announcement. This is the last trading day before the event date.
        before_prices = close[close.index < event_date]
        price_before = float(before_prices.iloc[-1]) if not before_prices.empty else None

        # Get closing price on the day or just after the announcement. This is the first trading day on or after the event date.
        after_open = close[close.index >= event_date]
        price_day0 = float(after_open.iloc[0]) if not after_open.empty else None

        # Get closing price 30 days after the announcement. This is the trading day closest to 30 days after the event date.
        price_day30 = float(after_open.iloc[-1]) if len(after_open) > 1 else None

        return price_before, price_day0, price_day30

    except Exception as e:
        # If there's an error (e.g., network issue, invalid ticker), print the error and skip
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None

# Loop through every row in our matched data set and pull stock prices. 
print("\nFetching stock prices... (this may take a few minutes)")

# Empty lists to store the fetched prices
price_before_list, price_day0_list, price_day30_list = [], [], []

for _, row in df_public.iterrows():
    pb, p0, p30 = get_price_window(row["ticker"], row["date"])
    price_before_list.append(pb)
    price_day0_list.append(p0)
    price_day30_list.append(p30)
    time.sleep(0.3) # Sleep for 100 milliseconds between requests

# Add the fetched prices to our DataFrame
df_public["price_before"] = price_before_list
df_public["price_day0"] = price_day0_list
df_public["price_day30"] = price_day30_list


# Engineering Return Columns
# Calculate immediate stock reaction (day of announcement vs day before)
df_public["return_immediate"] = (
    (df_public["price_day0"] - df_public["price_before"]) / df_public["price_before"] * 100  
)

# Calculate 30-Day Returns 
df_public["return_30day"] = (
    (df_public["price_day30"] - df_public["price_before"]) / df_public["price_before"] * 100
)
# Binary target for modeling: 1 = positive return 30 days after the layoff, and 0 = stock went down
df_public["target"] = (df_public["return_30day"] > 0).astype(int)

# Drop rows where we couldn't fetch stock prices (i.e., where price_before is null)
df_final = df_public.dropna(subset=["price_before", "price_day0", "price_day30"]).copy()

print(f"\nFinal Dataset Rows: {len(df_final)}")
print(df_final[["company", "ticker", "date", "total_laid_off", "return_immediate", "return_30day", "target"]].head(10))

# Save the cleaned and enriched dataset to a new CSV file for use in stage2
df_final.to_csv("layoffs_with_stocks.csv", index=False)
print("\nSaved to layoffs_with_stocks.csv")