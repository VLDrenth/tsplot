# Time Series Analysis Dashboard

A web-based tool for analyzing time series data with three analysis modes and various visualization options.

**Live Demo:** https://tsplot.streamlit.app/

## Features

**Analysis Modes:**
- Univariate Analysis - Visualize single or multiple time series
- Time Series Analysis - Compare a series with its shifted version for autocorrelation analysis
- Correlation Analysis - Analyze relationships between two different series with optional shifting

**Data Processing:**
- File upload support for CSV and Parquet files
- Data resampling (daily to weekly/monthly/quarterly/yearly) with multiple aggregation strategies
- Signal processing transforms: lowpass/highpass/bandpass filters, Savitzky-Golay smoothing, moving averages, exponential smoothing, detrending
- Date range filtering

**Visualizations:**
- Line, scatter, bar, and area plots for univariate data
- Binned scatter plots with configurable bin sizes
- Lag plots and autocorrelation functions for time series analysis
- Cross-correlation plots for correlation analysis
- Dual-axis time series plots for comparing different series

**Analysis Metrics:**
- Descriptive statistics for univariate data
- Correlation coefficients (Pearson, Spearman) and R² values
- Autocorrelation at specified lags

## Usage

1. Upload a CSV or Parquet file with time series data
2. Select the analysis mode
3. Choose time and value columns
4. Configure plot settings, transforms, and resampling as needed
5. View results and export data or plots

The app automatically detects datetime and numeric columns and provides appropriate options for each analysis mode.

## Requirements

- Python 3.10+
- streamlit
- pandas
- plotly
- numpy
- scipy