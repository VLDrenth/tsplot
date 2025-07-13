import numpy as np
import pandas as pd
from scipy import signal
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import uuid


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


def mom_percent(data: pd.Series) -> pd.Series:
    """Calculate Month-over-Month percentage change."""
    # Calculate percentage change from the same day/period one month ago
    # This works for both daily data (30-31 day lag) and monthly data (1 period lag)
    if isinstance(data.index, pd.DatetimeIndex):
        # For datetime index, shift by approximately 1 month
        result = data.pct_change(periods=30) * 100  # Approximate monthly shift for daily data
        
        # Try to detect frequency and adjust accordingly
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            if 'M' in inferred_freq or 'MS' in inferred_freq:  # Monthly data
                result = data.pct_change(periods=1) * 100
            elif 'Q' in inferred_freq:  # Quarterly data
                result = data.pct_change(periods=1) * 100  # Still month-over-month concept
            elif 'W' in inferred_freq:  # Weekly data
                result = data.pct_change(periods=4) * 100  # ~1 month = 4 weeks
        
    else:
        # For non-datetime index, assume appropriate lag based on data characteristics
        result = data.pct_change(periods=1) * 100
    
    return result


def yoy_percent(data: pd.Series) -> pd.Series:
    """Calculate Year-over-Year percentage change."""
    # Calculate percentage change from the same day/period one year ago
    if isinstance(data.index, pd.DatetimeIndex):
        # For datetime index, shift by approximately 1 year
        result = data.pct_change(periods=365) * 100  # Approximate yearly shift for daily data
        
        # Try to detect frequency and adjust accordingly
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            if 'M' in inferred_freq or 'MS' in inferred_freq:  # Monthly data
                result = data.pct_change(periods=12) * 100  # 12 months = 1 year
            elif 'Q' in inferred_freq:  # Quarterly data
                result = data.pct_change(periods=4) * 100   # 4 quarters = 1 year
            elif 'W' in inferred_freq:  # Weekly data
                result = data.pct_change(periods=52) * 100  # 52 weeks = 1 year
        
    else:
        # For non-datetime index, assume appropriate lag
        # Default to 12 periods (good for monthly data)
        result = data.pct_change(periods=12) * 100
    
    return result


def resample_timeseries(data: pd.DataFrame, time_col: str, value_cols: list, 
                       frequency: str = 'D', strategy: str = 'mean') -> pd.DataFrame:
    """
    Resample time series data to specified frequency.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with time series data
    time_col : str
        Name of the time column
    value_cols : list
        List of value columns to resample
    frequency : str
        Resampling frequency ('D', 'W', 'M', 'Q', 'Y')
    strategy : str
        Aggregation strategy ('mean', 'last', 'first', 'sum', 'max', 'min', 'count')
    
    Returns
    -------
    pd.DataFrame
        Resampled dataframe
    """
    # Make a copy and ensure time column is datetime
    df = data.copy()
    if df[time_col].dtype != 'datetime64[ns]':
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Set time column as index for resampling
    df_indexed = df.set_index(time_col)
    
    # Define aggregation strategy
    if strategy == 'mean':
        resampled = df_indexed[value_cols].resample(frequency).mean()
    elif strategy == 'last':
        resampled = df_indexed[value_cols].resample(frequency).last()
    elif strategy == 'first':
        resampled = df_indexed[value_cols].resample(frequency).first()
    elif strategy == 'sum':
        resampled = df_indexed[value_cols].resample(frequency).sum()
    elif strategy == 'max':
        resampled = df_indexed[value_cols].resample(frequency).max()
    elif strategy == 'min':
        resampled = df_indexed[value_cols].resample(frequency).min()
    elif strategy == 'count':
        resampled = df_indexed[value_cols].resample(frequency).count()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Reset index to get time column back
    resampled = resampled.reset_index()
    
    return resampled


@dataclass
class Transform:
    """Represents a single transform with its parameters."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transform_type: str = "none"
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def apply(self, data: pd.Series) -> pd.Series:
        """Apply this transform to the data."""
        if not self.enabled or self.transform_type == "none":
            return data
        return apply_transform(data, self.transform_type, **self.parameters)


class TransformPipeline:
    """Manages a pipeline of up to 3 transforms applied in sequence."""
    
    def __init__(self, max_transforms: int = 3):
        self.max_transforms = max_transforms
        self.transforms: List[Transform] = []
    
    def add_transform(self, transform_type: str, parameters: Dict[str, Any] = None) -> Transform:
        """Add a new transform to the pipeline."""
        if len(self.transforms) >= self.max_transforms:
            raise ValueError(f"Maximum {self.max_transforms} transforms allowed")
        
        transform = Transform(
            transform_type=transform_type,
            parameters=parameters or {}
        )
        self.transforms.append(transform)
        return transform
    
    def remove_transform(self, transform_id: str) -> bool:
        """Remove a transform by ID."""
        for i, transform in enumerate(self.transforms):
            if transform.id == transform_id:
                del self.transforms[i]
                return True
        return False
    
    def move_transform(self, transform_id: str, new_position: int) -> bool:
        """Move a transform to a new position."""
        if new_position < 0 or new_position >= len(self.transforms):
            return False
        
        # Find the transform
        transform_index = None
        for i, transform in enumerate(self.transforms):
            if transform.id == transform_id:
                transform_index = i
                break
        
        if transform_index is None:
            return False
        
        # Move the transform
        transform = self.transforms.pop(transform_index)
        self.transforms.insert(new_position, transform)
        return True
    
    def update_transform_parameters(self, transform_id: str, parameters: Dict[str, Any]) -> bool:
        """Update parameters for a specific transform."""
        for transform in self.transforms:
            if transform.id == transform_id:
                transform.parameters.update(parameters)
                return True
        return False
    
    def toggle_transform(self, transform_id: str) -> bool:
        """Enable/disable a specific transform."""
        for transform in self.transforms:
            if transform.id == transform_id:
                transform.enabled = not transform.enabled
                return True
        return False
    
    def apply(self, data: pd.Series) -> pd.Series:
        """Apply all enabled transforms in sequence."""
        result = data.copy()
        for transform in self.transforms:
            if transform.enabled:
                try:
                    result = transform.apply(result)
                except Exception as e:
                    print(f"Warning: Transform {transform.transform_type} failed: {e}")
                    continue
        return result
    
    def get_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all transforms in the pipeline."""
        return [
            {
                "id": t.id,
                "type": t.transform_type,
                "parameters": t.parameters,
                "enabled": t.enabled,
                "position": i
            }
            for i, t in enumerate(self.transforms)
        ]
    
    def clear(self):
        """Remove all transforms from the pipeline."""
        self.transforms.clear()
    
    def is_full(self) -> bool:
        """Check if the pipeline is at maximum capacity."""
        return len(self.transforms) >= self.max_transforms
    
    def is_empty(self) -> bool:
        """Check if the pipeline is empty."""
        return len(self.transforms) == 0


def get_transform_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available transforms and their parameters."""
    return {
        "lowpass": {
            "name": "Low-pass Filter",
            "description": "Removes high-frequency noise",
            "parameters": {
                "cutoff": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
                "order": {"type": "int", "min": 1, "max": 10, "default": 5}
            }
        },
        "highpass": {
            "name": "High-pass Filter", 
            "description": "Removes low-frequency trends",
            "parameters": {
                "cutoff": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01},
                "order": {"type": "int", "min": 1, "max": 10, "default": 5}
            }
        },
        "bandpass": {
            "name": "Band-pass Filter",
            "description": "Keeps frequencies in a specific range",
            "parameters": {
                "low": {"type": "float", "min": 0.01, "max": 0.4, "default": 0.05, "step": 0.01},
                "high": {"type": "float", "min": 0.1, "max": 0.5, "default": 0.2, "step": 0.01},
                "order": {"type": "int", "min": 1, "max": 10, "default": 5}
            }
        },
        "savgol": {
            "name": "Savitzky-Golay Filter",
            "description": "Smooths data while preserving features",
            "parameters": {
                "window_length": {"type": "int", "min": 5, "max": 51, "default": 11, "step": 2},
                "polyorder": {"type": "int", "min": 1, "max": 6, "default": 3}
            }
        },
        "moving_average": {
            "name": "Moving Average",
            "description": "Simple smoothing filter",
            "parameters": {
                "window": {"type": "int", "min": 3, "max": 100, "default": 10}
            }
        },
        "exponential_smoothing": {
            "name": "Exponential Smoothing",
            "description": "Weighted smoothing with exponential decay",
            "parameters": {
                "alpha": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.3, "step": 0.01}
            }
        },
        "detrend": {
            "name": "Detrend",
            "description": "Removes linear or constant trends",
            "parameters": {
                "method": {"type": "select", "options": ["linear", "constant"], "default": "linear"}
            }
        },
        "mom_percent": {
            "name": "MoM %",
            "description": "Month-over-Month percentage change (adapts to data frequency)",
            "parameters": {}
        },
        "yoy_percent": {
            "name": "YoY %", 
            "description": "Year-over-Year percentage change (adapts to data frequency)",
            "parameters": {}
        }
    }


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
    elif transform_type == "mom_percent":
        return mom_percent(data)
    elif transform_type == "yoy_percent":
        return yoy_percent(data)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
