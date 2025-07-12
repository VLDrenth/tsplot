import numpy as np
import pandas as pd
from scipy import signal
from typing import Union, Optional, Tuple


def lowpass_filter(data: pd.Series, cutoff: float, fs: float = 1.0, order: int = 5) -> pd.Series:
    """Apply Butterworth lowpass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, data.dropna())
    result = data.copy()
    result.loc[data.notna()] = filtered
    return result


def highpass_filter(data: pd.Series, cutoff: float, fs: float = 1.0, order: int = 5) -> pd.Series:
    """Apply Butterworth highpass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered = signal.filtfilt(b, a, data.dropna())
    result = data.copy()
    result.loc[data.notna()] = filtered
    return result


def bandpass_filter(data: pd.Series, low: float, high: float, fs: float = 1.0, order: int = 5) -> pd.Series:
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = signal.butter(order, [low_norm, high_norm], btype='band', analog=False)
    filtered = signal.filtfilt(b, a, data.dropna())
    result = data.copy()
    result.loc[data.notna()] = filtered
    return result


def savgol_filter(data: pd.Series, window_length: int, polyorder: int = 3) -> pd.Series:
    """Apply Savitzky-Golay filter."""
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    filtered = signal.savgol_filter(data.dropna(), window_length, polyorder)
    result = data.copy()
    result.loc[data.notna()] = filtered
    return result


def moving_average(data: pd.Series, window: int) -> pd.Series:
    """Apply simple moving average."""
    return data.rolling(window=window, center=True).mean()


def exponential_smoothing(data: pd.Series, alpha: float = 0.3) -> pd.Series:
    """Apply exponential smoothing."""
    return data.ewm(alpha=alpha).mean()


def detrend(data: pd.Series, method: str = 'linear') -> pd.Series:
    """Remove trend from data."""
    clean_data = data.dropna()
    if method == 'linear':
        detrended = signal.detrend(clean_data, type='linear')
    elif method == 'constant':
        detrended = signal.detrend(clean_data, type='constant')
    else:
        raise ValueError("Method must be 'linear' or 'constant'")
    
    result = data.copy()
    result.loc[data.notna()] = detrended
    return result


def apply_transform(data: pd.Series, transform_type: str, **kwargs) -> pd.Series:
    """Apply specified transform to data."""
    if transform_type == "none":
        return data
    elif transform_type == "lowpass":
        return lowpass_filter(data, **kwargs)
    elif transform_type == "highpass":
        return highpass_filter(data, **kwargs)
    elif transform_type == "bandpass":
        return bandpass_filter(data, **kwargs)
    elif transform_type == "savgol":
        return savgol_filter(data, **kwargs)
    elif transform_type == "moving_average":
        return moving_average(data, **kwargs)
    elif transform_type == "exponential_smoothing":
        return exponential_smoothing(data, **kwargs)
    elif transform_type == "detrend":
        return detrend(data, **kwargs)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
