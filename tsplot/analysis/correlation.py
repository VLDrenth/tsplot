import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional
from .base import BaseAnalysis
from .utils import shift_series, align_series, calculate_cross_correlation, calculate_correlation_metrics


class CorrelationAnalysis(BaseAnalysis):
    """Analysis for correlation between two different series."""
    
    def __init__(self, data: pd.DataFrame, time_col: str, x_col: str, y_col: str, shift: int = 0):
        super().__init__(data, time_col)
        self.x_col = x_col
        self.y_col = y_col
        self.shift = shift  # shift applied to y_col
        
    def get_required_columns(self) -> Dict[str, str]:
        return {
            'time_column': 'Required: Column containing timestamps',
            'x_column': 'Required: First numeric column for comparison',
            'y_column': 'Required: Second numeric column for comparison',
            'shift': 'Optional: Number of periods to shift second series'
        }
    
    def get_value_columns(self) -> List[str]:
        return [col for col in [self.x_col, self.y_col] if col]
    
    def validate_inputs(self, **kwargs) -> bool:
        return (self.x_col in self.data.columns and 
                self.y_col in self.data.columns and
                self.time_col in self.data.columns and
                isinstance(self.shift, int))
    
    def create_plot(self, plot_type: str, date_range: Optional[tuple] = None,
                   transform_pipeline = None, transform_type: str = "none", transform_params: Dict = None,
                   plot_params: Dict = None, show_markers: bool = False,
                   resample_params: Dict = None) -> go.Figure:
        """Create correlation analysis plot."""
        if transform_params is None:
            transform_params = {}
        if plot_params is None:
            plot_params = {}
            
        # Prepare data
        plot_df = self.prepare_data(date_range, resample_params)
        x_series = plot_df[self.x_col].copy()
        y_series = plot_df[self.y_col].copy()
        
        # Apply transforms
        if transform_pipeline is not None and not transform_pipeline.is_empty():
            x_series = self.apply_transforms(x_series, transform_pipeline=transform_pipeline)
            y_series = self.apply_transforms(y_series, transform_pipeline=transform_pipeline)
        elif transform_type != "none":
            x_series = self.apply_transforms(x_series, transform_type=transform_type, **transform_params)
            y_series = self.apply_transforms(y_series, transform_type=transform_type, **transform_params)
        
        # Apply shift to y_series
        if self.shift != 0:
            y_series = shift_series(y_series, self.shift)
        
        # Create plot based on type
        fig = go.Figure()
        
        if plot_type == "Time Series":
            # Plot both series over time
            fig.add_trace(go.Scatter(
                x=plot_df[self.time_col],
                y=x_series,
                mode='lines+markers' if show_markers else 'lines',
                name=self.x_col,
                line=dict(color='blue')
            ))
            
            y_name = f"{self.y_col} (shift {self.shift})" if self.shift != 0 else self.y_col
            fig.add_trace(go.Scatter(
                x=plot_df[self.time_col],
                y=y_series,
                mode='lines+markers' if show_markers else 'lines',
                name=y_name,
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"Time Series Comparison: {self.x_col} vs {self.y_col}",
                xaxis_title=self.time_col,
                yaxis=dict(title=self.x_col, side='left', color='blue'),
                yaxis2=dict(title=y_name, side='right', overlaying='y', color='red')
            )
            
        elif plot_type == "Scatter":
            # Correlation scatter plot
            x_aligned, y_aligned = align_series(x_series, y_series, method='inner')
            valid_mask = x_aligned.notna() & y_aligned.notna()
            x_vals = x_aligned[valid_mask]
            y_vals = y_aligned[valid_mask]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(size=plot_params.get('marker_size', 6)),
                name='Data points'
            ))
            
            # Add trend line if there are enough points
            if len(x_vals) > 1:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend line',
                    showlegend=False
                ))
            
            y_title = f"{self.y_col} (shift {self.shift})" if self.shift != 0 else self.y_col
            fig.update_layout(
                title=f"Correlation Plot: {self.x_col} vs {self.y_col}",
                xaxis_title=self.x_col,
                yaxis_title=y_title
            )
            
        elif plot_type == "Cross Correlation":
            # Cross-correlation function
            max_lags = min(50, len(x_series) // 4)
            lags, cross_corr = calculate_cross_correlation(x_series, y_series, max_lags)
            
            fig.add_trace(go.Scatter(
                x=lags,
                y=cross_corr,
                mode='lines+markers',
                name='Cross-correlation',
                line=dict(color='green')
            ))
            
            # Add significance bounds
            n = len(x_series.dropna())
            confidence_bound = 1.96 / np.sqrt(n)
            fig.add_hline(y=confidence_bound, line_dash="dash", line_color="red",
                         annotation_text="95% confidence")
            fig.add_hline(y=-confidence_bound, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title=f"Cross-Correlation: {self.x_col} vs {self.y_col}",
                xaxis_title="Lag",
                yaxis_title="Cross-correlation"
            )
        
        return fig
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate correlation analysis metrics."""
        df = self.prepare_data()
        x_series = df[self.x_col].copy()
        y_series = df[self.y_col].copy()
        
        # Apply shift to y_series
        if self.shift != 0:
            y_series = shift_series(y_series, self.shift)
        
        # Align series and calculate metrics
        x_aligned, y_aligned = align_series(x_series, y_series, method='inner')
        correlation_metrics = calculate_correlation_metrics(x_aligned, y_aligned)
        
        # Calculate cross-correlation to find optimal lag
        try:
            max_lags = min(20, len(x_series) // 4)
            lags, cross_corr = calculate_cross_correlation(x_series, df[self.y_col], max_lags)
            optimal_lag_idx = np.argmax(np.abs(cross_corr))
            optimal_lag = lags[optimal_lag_idx]
            max_cross_corr = cross_corr[optimal_lag_idx]
        except:
            optimal_lag = 0
            max_cross_corr = np.nan
        
        return {
            'shift_applied': self.shift,
            'optimal_lag': optimal_lag,
            'max_cross_correlation': max_cross_corr,
            **correlation_metrics,
            'x_series_length': len(x_series.dropna()),
            'y_series_length': len(df[self.y_col].dropna())
        }