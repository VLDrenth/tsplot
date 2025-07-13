import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from .base import BaseAnalysis
from ..plot_types.bin_scatter import bin_scatter


class UnivariateAnalysis(BaseAnalysis):
    """Analysis for single time series."""
    
    def __init__(self, data: pd.DataFrame, time_col: str, value_cols: List[str]):
        super().__init__(data, time_col)
        self.value_cols = value_cols
        
    def get_required_columns(self) -> Dict[str, str]:
        return {
            'time_column': 'Required: Column containing timestamps',
            'value_columns': 'Required: One or more numeric columns to analyze'
        }
    
    def get_value_columns(self) -> List[str]:
        return self.value_cols
    
    def validate_inputs(self, **kwargs) -> bool:
        return bool(self.value_cols) and self.time_col in self.data.columns
    
    def create_plot(self, plot_type: str, date_range: Optional[tuple] = None, 
                   transform_pipeline = None, transform_type: str = "none", transform_params: Dict = None,
                   plot_params: Dict = None, show_markers: bool = False,
                   resample_params: Dict = None) -> go.Figure:
        """Create univariate time series plot."""
        if transform_params is None:
            transform_params = {}
        if plot_params is None:
            plot_params = {}
            
        # Prepare data
        plot_df = self.prepare_data(date_range, resample_params)
        plot_df = plot_df[[self.time_col] + self.value_cols].copy()
        
        # Apply transforms
        if transform_pipeline is not None and not transform_pipeline.is_empty():
            for col in self.value_cols:
                try:
                    plot_df[col] = self.apply_transforms(plot_df[col], transform_pipeline=transform_pipeline)
                except Exception:
                    continue
        elif transform_type != "none":
            for col in self.value_cols:
                try:
                    plot_df[col] = self.apply_transforms(plot_df[col], transform_type=transform_type, **transform_params)
                except Exception:
                    continue
        
        # Create figure
        fig = go.Figure()
        
        # Add traces based on plot type
        if plot_type == "Bin Scatter":
            for col in self.value_cols:
                bin_fig = bin_scatter(
                    plot_df, 
                    self.time_col, 
                    col, 
                    bin_size=plot_params.get('bin_size', 10),
                    plot_type=plot_params.get('bin_plot_type', 'scatter')
                )
                for trace in bin_fig.data:
                    trace.name = f"{col} (binned)"
                    fig.add_trace(trace)
        else:
            for col in self.value_cols:
                if plot_type == "Line":
                    scatter_trace = go.Scatter(
                        x=plot_df[self.time_col],
                        y=plot_df[col],
                        mode='lines+markers' if show_markers else 'lines',
                        name=col
                    )
                    if 'line_smoothing' in plot_params and plot_params['line_smoothing'] > 0:
                        scatter_trace.line = dict(smoothing=plot_params['line_smoothing'])
                    fig.add_trace(scatter_trace)
                elif plot_type == "Scatter":
                    fig.add_trace(go.Scatter(
                        x=plot_df[self.time_col],
                        y=plot_df[col],
                        mode='markers',
                        marker=dict(size=plot_params.get('marker_size', 6)),
                        name=col
                    ))
                elif plot_type == "Bar":
                    fig.add_trace(go.Bar(
                        x=plot_df[self.time_col],
                        y=plot_df[col],
                        width=plot_params.get('bar_width', 0.8),
                        name=col
                    ))
                elif plot_type == "Area":
                    fig.add_trace(go.Scatter(
                        x=plot_df[self.time_col],
                        y=plot_df[col],
                        mode='lines',
                        fill='tozeroy',
                        name=col
                    ))
        
        # Update layout
        fig.update_layout(
            title=f"{plot_type} Chart: {', '.join(self.value_cols)}",
            xaxis_title=self.time_col,
            yaxis_title="Values",
            hovermode='x unified',
            showlegend=len(self.value_cols) > 1
        )
        
        return fig
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate univariate statistics."""
        metrics = {}
        df = self.prepare_data()
        
        for col in self.value_cols:
            series = df[col].dropna()
            if len(series) > 0:
                metrics[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis()
                }
        
        return metrics