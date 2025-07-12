import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional
from .base import BaseAnalysis
from .utils import shift_series, calculate_autocorrelation, calculate_correlation_metrics


class TimeSeriesAnalysis(BaseAnalysis):
    """Analysis for time series vs its shifted version."""
    
    def __init__(self, data: pd.DataFrame, time_col: str, value_col: str, lag: int = 1):
        super().__init__(data, time_col)
        self.value_col = value_col
        self.lag = lag
        
    def get_required_columns(self) -> Dict[str, str]:
        return {
            'time_column': 'Required: Column containing timestamps',
            'value_column': 'Required: Single numeric column to analyze',
            'lag': 'Required: Number of periods to shift for comparison'
        }
    
    def get_value_columns(self) -> List[str]:
        return [self.value_col] if self.value_col else []
    
    def validate_inputs(self, **kwargs) -> bool:
        return (self.value_col in self.data.columns and 
                self.time_col in self.data.columns and
                isinstance(self.lag, int))
    
    def create_plot(self, plot_type: str, date_range: Optional[tuple] = None,
                   transform_type: str = "none", transform_params: Dict = None,
                   plot_params: Dict = None, show_markers: bool = False,
                   resample_params: Dict = None) -> go.Figure:
        """Create time series lag analysis plot."""
        if transform_params is None:
            transform_params = {}
        if plot_params is None:
            plot_params = {}
            
        # Prepare data
        plot_df = self.prepare_data(date_range, resample_params)
        series = plot_df[self.value_col].copy()
        
        # Apply transforms
        if transform_type != "none":
            series = self.apply_transforms(series, transform_type, **transform_params)
        
        # Create shifted version
        shifted_series = shift_series(series, self.lag)
        
        # Create plot based on type
        fig = go.Figure()
        
        if plot_type == "Time Series":
            # Plot original and shifted series over time
            fig.add_trace(go.Scatter(
                x=plot_df[self.time_col],
                y=series,
                mode='lines+markers' if show_markers else 'lines',
                name=f'{self.value_col} (original)',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_df[self.time_col],
                y=shifted_series,
                mode='lines+markers' if show_markers else 'lines',
                name=f'{self.value_col} (lag {self.lag})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Time Series Comparison: {self.value_col} vs Lag {self.lag}",
                xaxis_title=self.time_col,
                yaxis_title="Values"
            )
            
        elif plot_type == "Lag Plot":
            # Scatter plot of series vs shifted version
            valid_mask = series.notna() & shifted_series.notna()
            x_vals = series[valid_mask]
            y_vals = shifted_series[valid_mask]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(size=plot_params.get('marker_size', 6)),
                name=f'Lag {self.lag} Plot'
            ))
            
            # Add diagonal line for reference
            min_val = min(x_vals.min(), y_vals.min())
            max_val = max(x_vals.max(), y_vals.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='y=x reference',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"Lag Plot: {self.value_col} vs Lag {self.lag}",
                xaxis_title=f"{self.value_col} (t)",
                yaxis_title=f"{self.value_col} (t-{self.lag})"
            )
            
        elif plot_type == "Autocorrelation":
            # Autocorrelation function plot
            max_lags = min(50, len(series) // 4)
            lags, autocorr = calculate_autocorrelation(series, max_lags)
            
            fig.add_trace(go.Scatter(
                x=lags,
                y=autocorr,
                mode='lines+markers',
                name='Autocorrelation',
                line=dict(color='blue')
            ))
            
            # Add significance bounds (95% confidence)
            n = len(series.dropna())
            confidence_bound = 1.96 / np.sqrt(n)
            fig.add_hline(y=confidence_bound, line_dash="dash", line_color="red", 
                         annotation_text="95% confidence")
            fig.add_hline(y=-confidence_bound, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f"Autocorrelation Function: {self.value_col}",
                xaxis_title="Lag",
                yaxis_title="Autocorrelation"
            )
        
        return fig
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate time series analysis metrics."""
        df = self.prepare_data()
        series = df[self.value_col].dropna()
        shifted_series = shift_series(series, self.lag)
        
        # Calculate correlation between original and shifted
        correlation_metrics = calculate_correlation_metrics(series, shifted_series)
        
        # Calculate autocorrelation at specific lag
        if len(series) > abs(self.lag):
            autocorr_at_lag = series.autocorr(lag=abs(self.lag))
        else:
            autocorr_at_lag = np.nan
        
        return {
            'lag': self.lag,
            'autocorr_at_lag': autocorr_at_lag,
            **correlation_metrics,
            'series_length': len(series),
            'valid_pairs': correlation_metrics['n_points']
        }