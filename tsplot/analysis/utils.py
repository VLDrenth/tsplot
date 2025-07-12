import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from scipy import stats


def shift_series(series: pd.Series, lag: int, fill_value=np.nan) -> pd.Series:
    """Shift a time series by specified lag periods."""
    if lag == 0:
        return series.copy()
    elif lag > 0:
        # Positive lag: shift forward (delay)
        return series.shift(lag, fill_value=fill_value)
    else:
        # Negative lag: shift backward (advance)
        return series.shift(lag, fill_value=fill_value)


def align_series(s1: pd.Series, s2: pd.Series, method: str = 'inner') -> Tuple[pd.Series, pd.Series]:
    """Align two series by their indices."""
    if method == 'inner':
        # Keep only common indices
        common_idx = s1.index.intersection(s2.index)
        return s1.loc[common_idx], s2.loc[common_idx]
    elif method == 'outer':
        # Keep all indices, fill missing with NaN
        all_idx = s1.index.union(s2.index)
        s1_aligned = s1.reindex(all_idx)
        s2_aligned = s2.reindex(all_idx)
        return s1_aligned, s2_aligned
    else:
        raise ValueError("Method must be 'inner' or 'outer'")


def calculate_correlation_metrics(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """Calculate correlation metrics between two series."""
    # Remove NaN values for calculations
    valid_mask = x.notna() & y.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        return {
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'r_squared': np.nan,
            'n_points': len(x_clean)
        }
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
    
    # R-squared
    r_squared = pearson_r ** 2
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r_squared': r_squared,
        'n_points': len(x_clean)
    }


def calculate_autocorrelation(series: pd.Series, max_lags: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate autocorrelation function using numpy."""
    # Remove NaN values
    clean_series = series.dropna().values
    
    if len(clean_series) < max_lags + 1:
        max_lags = max(1, len(clean_series) - 1)
    
    # Calculate autocorrelation manually
    autocorr = []
    n = len(clean_series)
    mean = np.mean(clean_series)
    var = np.var(clean_series)
    
    for lag in range(max_lags + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            if n - lag > 0:
                c = np.mean((clean_series[:-lag] - mean) * (clean_series[lag:] - mean))
                autocorr.append(c / var)
            else:
                autocorr.append(0.0)
    
    lags = np.arange(len(autocorr))
    
    return lags, np.array(autocorr)


def calculate_cross_correlation(x: pd.Series, y: pd.Series, max_lags: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate cross-correlation between two series."""
    # Align series and remove NaN values
    x_aligned, y_aligned = align_series(x, y, method='inner')
    valid_mask = x_aligned.notna() & y_aligned.notna()
    x_clean = x_aligned[valid_mask].values
    y_clean = y_aligned[valid_mask].values
    
    if len(x_clean) < 2:
        return np.array([0]), np.array([0])
    
    # Calculate cross-correlation using numpy
    correlation = np.correlate(x_clean, y_clean, mode='full')
    
    # Normalize
    correlation = correlation / (len(x_clean) * np.std(x_clean) * np.std(y_clean))
    
    # Create lag array
    lags = np.arange(-len(x_clean) + 1, len(x_clean))
    
    # Limit to requested max_lags
    if max_lags < len(lags) // 2:
        center = len(lags) // 2
        start = center - max_lags
        end = center + max_lags + 1
        lags = lags[start:end]
        correlation = correlation[start:end]
    
    return lags, correlation