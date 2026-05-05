import pandas as pd

# Load Central Ticker Map 
TICKER_MAP = {
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

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by dropping rows with missing values and resetting the index.
    
    Steps: 
        - Convert 'date' column to datetime
        - Drop Rows with Missing Values (company, date, total laid off)
        - Filter to US companies only (if 'location' column exists)
        - Reset Index 
    
    Parameters:
        df : raw pandas DataFrame loaded from layoffs.csv
        
    Returns:
        Cleaned pandas Dataframe 
    """
    # Convert column from string to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce") 

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=["company", "date", "total_laid_off"])

    # Filter to US Based Companies Only
    df = df[df["country"] == "United States"].copy()

    # Reset index so row numbers are sequential after dropping rows
    df = df.reset_index(drop=True)

    return df

def map_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map company names to their stock tickers using a predefined mapping.
    Rows without matching ticker are dropped 
    
    Parameters:
        df : pandas DataFrame with a 'company' column
        
    Returns:
        pandas DataFrame with only publicly matched companies and a 'ticker' column 
    """
    df["ticker"] = df["company"].map(TICKER_MAP)
    
    # Drop rows where ticker mapping is not found (i.e., company not in TICKER_MAP)
    df = df.dropna(subset=["ticker"]).copy().reset_index(drop=True)

    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage returns for stock prices before, on, and after the layoff announcement.
    Assumes dataframe has 'price_before', 'price_day0', and 'price_day30' columns with stock prices.
    
    Adds:
        - 'return_immediate': Percentage return from price_before to price_day0
        - 'return_day30': Percentage return from price_before to price_day30
        - 'target' : 1 if stock was up in 30 days, 0 if down or unchanged
    
    Parameters:
        df : pandas DataFrame with stock price columns
        
    Returns:
        Dataframe with return column added 
    """
    # Immediate Return day vs day before
    df["return_immediate"] = ((df["price_day0"] - df["price_before"]) / df["price_before"] * 100)

    # Return after 30 days vs day before
    df["return_30day"] = ((df["price_day30"] - df["price_before"]) / df["price_before"] * 100)

    # Create binary target variable: 1 if stock was up after 30 days, 0 if down or unchanged
    df["target"] = (df["return_30day"] > 0).astype(int)
    
    return df

def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter the Dataframe to a specific date range.

    Parameters:
        df : pandas DataFrame with a 'date' column
        start_date : string in 'YYYY-MM-DD' format representing the start of the date range
        end_date : string in 'YYYY-MM-DD' format representing the end of the date range
    Returns:
        Filtered Dataframe 
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()
    
def filter_by_industry(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    """
    Filter the DataFrame to a specific industry.

    Parameters:
        df : pandas DataFrame with an 'industry' column
        industries : list of industry strings to keep

    Returns:
        Filtered DataFrame 
    """
    # If the list is empty or None, return the full Dataframe unchanged 
    if not industry:
        return df
    return df[df["industry"].isin(industry)].copy()

def filter_by_company(df: pd.DataFrame, company: str) -> pd.DataFrame:
    """
    Filter the DataFrame to show matching rows of a specific company.

    Parameters:
        df : pandas DataFrame with a 'company' column
        companies : list of company name strings to keep 

    Returns:
        Filtered DataFrame 
    """
    if not company:
        return df
    return df[df["company"].isin(company)].copy()

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get key summary statistics from layoffs dataset. 

    Parameters:
        df : enriched dataframe with return columns (return_immediate, return_30day)

    Returns:
        Dictionary with summary statistics for the return columns
    """
    if df.empty:
        return {}
    
    # Find Company with Largest single layoff event
    max_idx = df["total_laid_off"].idxmax()
    best_idx = df["return_30day"].idxmax()
    worst_idx = df["return_30day"].idxmin()

    return {
        "total_events": len(df),
        "total_laid_off": int(df["total_laid_off"].sum()),
        "avg_immediate_return": round(df["return_immediate"].mean(), 2),
        "avg_30day_return": round(df["return_30day"].mean(), 2),
        "pct_stock_up" : round(df["target"].mean() * 100, 1),
        "largest_layoff_co" : df.loc[max_idx, "company"],
        "largest_layoff_n" : int(df.loc[max_idx, "total_laid_off"]),
        "best_return_co" : df.loc[best_idx, "company"],
        "best_return_val" : round(df.loc[best_idx, "return_30day"], 1),
        "worst_return_co" : df.loc[worst_idx, "company"],
        "worst_return_val" : round(df.loc[worst_idx, "return_30day"], 1),
    }