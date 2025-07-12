from abc import ABC, abstractmethod
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from ..transforms import apply_transform, resample_timeseries


class BaseAnalysis(ABC):
    """Base class for all analysis types."""
    
    def __init__(self, data: pd.DataFrame, time_col: str):
        self.data = data.copy()
        self.time_col = time_col
        self.transforms = {}
        self.plot_params = {}
        
    def apply_transforms(self, series: pd.Series, transform_type: str = "none", **transform_params) -> pd.Series:
        """Apply transforms to a series."""
        if transform_type == "none":
            return series
        return apply_transform(series, transform_type, **transform_params)
    
    def prepare_data(self, date_range: Optional[Tuple] = None, resample_params: Optional[Dict] = None) -> pd.DataFrame:
        """Prepare and filter data for analysis."""
        df = self.data.copy()
        
        # Convert time column to datetime if needed
        if df[self.time_col].dtype != 'datetime64[ns]':
            df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # Apply date filtering if provided
        if date_range and len(date_range) == 2:
            mask = (df[self.time_col] >= pd.to_datetime(date_range[0])) & \
                   (df[self.time_col] <= pd.to_datetime(date_range[1]))
            df = df[mask].copy()
        
        # Sort by time
        df = df.sort_values(self.time_col)
        
        # Apply resampling if requested
        if resample_params and len(resample_params) > 0:
            # Get value columns for this analysis type
            value_cols = self.get_value_columns()
            if value_cols:
                df = resample_timeseries(
                    df, 
                    self.time_col, 
                    value_cols,
                    frequency=resample_params.get('frequency', 'D'),
                    strategy=resample_params.get('strategy', 'mean')
                )
        
        return df
    
    @abstractmethod
    def get_required_columns(self) -> Dict[str, str]:
        """Return required column specifications for this analysis type."""
        pass
    
    @abstractmethod
    def get_value_columns(self) -> List[str]:
        """Return list of value columns for this analysis type."""
        pass
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate that all required inputs are provided."""
        pass
    
    @abstractmethod
    def create_plot(self, plot_type: str, **kwargs) -> go.Figure:
        """Create the plot for this analysis type."""
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate relevant statistical metrics."""
        pass