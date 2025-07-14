import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from .analysis import UnivariateAnalysis, TimeSeriesAnalysis, CorrelationAnalysis


class DataProcessor:
    """Unified data processing for all analysis modes."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._detect_column_types()
    
    def _detect_column_types(self):
        """Detect datetime and numeric columns."""
        self.datetime_cols = []
        self.numeric_cols = []
        
        for col in self.data.columns:
            # Check if already datetime type or can be converted
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_cols.append(col)
            elif self.data[col].dtype == 'object':
                try:
                    pd.to_datetime(self.data[col])
                    self.datetime_cols.append(col)
                except:
                    pass
            elif pd.api.types.is_numeric_dtype(self.data[col]):
                self.numeric_cols.append(col)
    
    def get_column_options(self, column_type: str) -> List[str]:
        """Get available columns of specified type."""
        if column_type == 'datetime':
            return self.datetime_cols if self.datetime_cols else self.data.columns.tolist()
        elif column_type == 'numeric':
            return self.numeric_cols if self.numeric_cols else self.data.columns.tolist()
        else:
            return self.data.columns.tolist()
    
    def create_analysis(self, mode: str, **params) -> Union[UnivariateAnalysis, TimeSeriesAnalysis, CorrelationAnalysis]:
        """Create appropriate analysis object based on mode."""
        time_col = params.get('time_col')
        
        if mode == "Univariate Analysis":
            value_cols = params.get('value_cols', [])
            return UnivariateAnalysis(self.data, time_col, value_cols)
            
        elif mode == "Time Series Analysis":
            value_col = params.get('value_col')
            lag = params.get('lag', 1)
            return TimeSeriesAnalysis(self.data, time_col, value_col, lag)
            
        elif mode == "Correlation Analysis":
            x_col = params.get('x_col')
            y_col = params.get('y_col')
            shift = params.get('shift', 0)
            return CorrelationAnalysis(self.data, time_col, x_col, y_col, shift)
            
        else:
            raise ValueError(f"Unknown analysis mode: {mode}")
    
    def validate_analysis_params(self, mode: str, **params) -> Tuple[bool, str]:
        """Validate parameters for specified analysis mode."""
        time_col = params.get('time_col')
        
        if not time_col or time_col not in self.data.columns:
            return False, "Time column is required and must exist in data"
        
        if mode == "Univariate Analysis":
            value_cols = params.get('value_cols', [])
            if not value_cols:
                return False, "At least one value column is required"
            missing_cols = [col for col in value_cols if col not in self.data.columns]
            if missing_cols:
                return False, f"Missing columns: {missing_cols}"
                
        elif mode == "Time Series Analysis":
            value_col = params.get('value_col')
            if not value_col or value_col not in self.data.columns:
                return False, "Value column is required and must exist in data"
            lag = params.get('lag', 1)
            if not isinstance(lag, int):
                return False, "Lag must be an integer"
                
        elif mode == "Correlation Analysis":
            x_col = params.get('x_col')
            y_col = params.get('y_col')
            if not x_col or x_col not in self.data.columns:
                return False, "X column is required and must exist in data"
            if not y_col or y_col not in self.data.columns:
                return False, "Y column is required and must exist in data"
            if x_col == y_col:
                return False, "X and Y columns must be different"
            shift = params.get('shift', 0)
            if not isinstance(shift, int):
                return False, "Shift must be an integer"
        
        return True, "Parameters are valid"
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'datetime_columns': self.datetime_cols,
            'numeric_columns': self.numeric_cols,
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict()
        }