import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from transform import (
    filter_by_date,
    filter_by_industry,
    filter_by_company,
    get_summary_stats
)

# Page Configuration
st.set_page_config(
    page_title="Layoffs and Leverage",
    page_icon="📉",
    layout="wide"
)

# Header 
st.title("📉 Layoffs and Leverage")
st.markdown("*Analyzing how tech layoff announcements affect stock market performance*")
st.divider()

# Load Data 
@st.cache_data
def load_data():
    """Load the enriched layoffs dataset produced by phase1_cleaning.py"""
    df = pd.read_csv("layoffs_with_stocks.csv", parse_dates=["date"])
    return df

df_raw = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("Use the filters below to explore specific segments of the data.")

# Date Range Filter
min_date = df_raw["date"].min().date()
max_date = df_raw["date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Data Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Industry Filter
all_industries = sorted(df_raw["industry"].dropna().unique().tolist())
selected_industries = st.sidebar.multiselect(
    "Industry",
    options=all_industries,
    default=[],
    placeholder="All industries"
)

# Company Filter
all_companies = sorted(df_raw["company"].dropna().unique().tolist())
selected_companies = st.sidebar.multiselect(
    "Company",
    options=all_companies,
    default=[],
    placeholder="All companies"
)

# Apply Filters using transform.py functions
df = filter_by_date(df_raw, str(start_date), str(end_date))
df = filter_by_industry(df, selected_industries)
df = filter_by_company(df, selected_companies)

# Show How Many Rows are Currently Visible After Filtering
st.sidebar.markdown(f"**{len(df)}** rows match the current filters")

# Handle Empty Filter Results
if df.empty:
    st.warning("No data matches the current filters. Please adjust your filters to see results.")
    st.stop()

# KPI Summary Stats (from transform.py)
stats = get_summary_stats(df)

st.subheader("📊 Key Summary Statistics")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Layoff Events", f"{stats['total_events']}")
col2.metric("Total Laid Off", f"{stats['total_laid_off']}")
col3.metric("Avg Immediate Return", f"{stats['avg_immediate_return']}%")
col4.metric("Avg 30-Day Return", f"{stats['avg_30day_return']}%")  
col5.metric("% Stocks Up (30 Day)", f"{stats['pct_stock_up']}%")

st.divider()

# Charts 
st.subheader("📈 Visualizations")

# Helper to Set Consistent Plot Style 
def set_style(ax):
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, color="#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Layoffs Over Time + Top Companies
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Layoffs Over Time**")
    
    # Group by Month and sum total_laid_off
    df["month"] = df["date"].dt.to_period("M")
    by_month = df.groupby("month")["total_laid_off"].sum().reset_index()
    by_month["month"] = by_month["month"].astype(str)  # Convert back to string for plotting

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_month["month"], by_month["total_laid_off"], color="#e63946", edgecolor="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Employees Laid Off")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right", fontsize=7)
    set_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    st.markdown("**Top 10 Companies by Total Laid Off**")
    
    top_co = (
        df.groupby("company")["total_laid_off"]
        .sum()
        .sort_values(ascending=True)
        .tail(10)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(top_co.index, top_co.values, color="#e63946", edgecolor="white")
    for bar, val in zip(bars, top_co.values):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2, f"{int(val):,}", va="center", fontsize=8)
    ax.set_xlabel("Total Laid Off")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    set_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Avg Returns + Return Distribution
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**Average Immediate vs. 30-Day Returns**")

    avg_vals = [df["return_immediate"].mean(), df["return_30day"].mean()]
    labels = ["Immediate\n(Day of)", "30-Day"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, avg_vals, color=["#457b9d", "#2a9d8f"], edgecolor="white")
    for bar, val in zip(bars, avg_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if val >= 0 else -0.4), f"{val:.2f}%", ha="center", fontweight="bold", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Average Return (%)")
    set_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_d:
    st.markdown("**Distribution of 30-Day Returns**")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["return_30day"].dropna(), bins=25, color="#457b9d", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="0%")
    ax.axvline(df["return_30day"].mean(), color="orange", linestyle="--", linewidth=1.2, label=f"Mean: {df['return_30day'].mean():.1f}%")
    ax.set_xlabel("30-Day Return (%)")
    ax.set_ylabel("Count")
    ax.legend()
    set_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Industry Returns + Scatter
col_e, col_f = st.columns(2)

with col_e:
    st.markdown("**Average 30-Day Return by Industry**")

    industry_returns = df.groupby("industry")["return_30day"].mean().dropna().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2a9d8f" if v > 0 else "#e63946" for v in industry_returns.values]
    ax.barh(industry_returns.index, industry_returns.values, color=colors, edgecolor="white")   
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average 30-Day Return (%)")
    set_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_f:
    st.markdown("**Layoff Size vs. 30-Day Return**")

    colors_scatter = df["target"].map({1: "#2a9d8f", 0: "#e63946"})  # Green for positive return, Red for negative/zero

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["total_laid_off"], df["return_30day"], alpha=0.7, color=colors_scatter, edgecolor="white", linewidth=0.5, s=60)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Total Laid Off")
    ax.set_ylabel("30-Day Return (%)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    legend_els = [
        Patch(facecolor="#2a9d8f", label="Stock Up"),
        Patch(facecolor="#e63946",label="Stock Down")
    ]
    ax.legend(handles=legend_els)
    set_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# Raw Data Table
st.subheader("📋 Raw Data")

# Show filtered data in a table with key columns
display_cols = ["company", "ticker", "industry", "date", "total_laid_off", "percentage_laid_off", "return_immediate", "return_30day", "target"]
st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True)

# Download Button for ability to export filtered data as a csv
csv = df[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="layoffs_filtered.csv",
    mime="text/csv"
)

st.divider()

# Footer
st.markdown("Made with ❤️ by [Sachin Sastry] | Data Source: [Layoffs Dataset](https://www.kaggle.com/datasets/benedekrozemberczki/layoffs) | Stock Data and Prices via yfinance")