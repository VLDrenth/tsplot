import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

from tsplot.plot_types.bin_scatter import bin_scatter
from tsplot.transforms import apply_transform
from tsplot.data_processor import DataProcessor

st.set_page_config(page_title="Time Series Dashboard", layout="wide")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None

# File upload
if not st.session_state.file_uploaded:
    st.title("Time Series Visualization Dashboard")
    st.markdown("### Upload your data file")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Parquet file",
        type=['csv', 'parquet'],
        help="Upload a time series dataset"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # parquet
                df = pd.read_parquet(uploaded_file)
            
            # Store in session state
            st.session_state.data = df
            st.session_state.data_processor = DataProcessor(df)
            st.session_state.file_uploaded = True
            st.session_state.filename = uploaded_file.name
            st.rerun()
            
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Main dashboard
else:
    df = st.session_state.data
    processor = st.session_state.data_processor
    
    # Sidebar for settings
    with st.sidebar:
        st.title("Dashboard Settings")
        
        # Analysis mode selector
        st.markdown("### Analysis Mode")
        analysis_mode = st.selectbox(
            "Select analysis type",
            ["Univariate Analysis", "Time Series Analysis", "Correlation Analysis"],
            help="Choose the type of analysis to perform"
        )
        
        st.markdown("---")
        

        # Column selection (mode-aware)
        st.markdown("### Column Selection")
        
        # Get column options
        datetime_cols = processor.get_column_options('datetime')
        numeric_cols = processor.get_column_options('numeric')
        
        # Time column selection (common to all modes)
        time_col = st.selectbox(
            "Select time column",
            options=datetime_cols,
            help="Select the column containing timestamps"
        )
        
        # Mode-specific column selection
        if analysis_mode == "Univariate Analysis":
            value_cols = st.multiselect(
                "Select value columns to plot",
                options=numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
                help="Select one or more numeric columns to visualize"
            )
            x_col = None
            y_col = None
            lag = 0
            shift = 0
            
        elif analysis_mode == "Time Series Analysis":
            value_col = st.selectbox(
                "Select value column",
                options=numeric_cols,
                help="Select the numeric column to analyze"
            )
            lag = st.number_input(
                "Lag periods",
                min_value=1,
                max_value=100,
                value=1,
                help="Number of periods to shift for comparison"
            )
            value_cols = [value_col] if value_col else []
            x_col = None
            y_col = value_col
            shift = lag
            
        elif analysis_mode == "Correlation Analysis":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox(
                    "Select X column",
                    options=numeric_cols,
                    help="First variable for correlation analysis"
                )
            with col2:
                y_col = st.selectbox(
                    "Select Y column", 
                    options=numeric_cols,
                    index=1 if len(numeric_cols) > 1 else 0,
                    help="Second variable for correlation analysis"
                )
            
            shift = st.number_input(
                "Shift Y column",
                min_value=-50,
                max_value=50,
                value=0,
                help="Number of periods to shift Y column (negative = advance, positive = delay)"
            )
            value_cols = [x_col, y_col] if x_col and y_col else []
        
        st.markdown("---")
        
        # Plot settings (mode-aware)
        st.markdown("### Plot Settings")
        
        # Plot type options based on analysis mode
        if analysis_mode == "Univariate Analysis":
            plot_options = ["Line", "Scatter", "Bar", "Area", "Bin Scatter"]
        elif analysis_mode == "Time Series Analysis":
            plot_options = ["Time Series", "Lag Plot", "Autocorrelation"]
        elif analysis_mode == "Correlation Analysis":
            plot_options = ["Time Series", "Scatter", "Cross Correlation"]
        
        plot_type = st.selectbox(
            "Chart type",
            plot_options,
            help="Select the type of chart"
        )
        
        show_markers = st.checkbox("Show markers", value=False)
        
        plot_height = st.slider(
            "Plot height (px)",
            min_value=300,
            max_value=800,
            value=500,
            step=50
        )
        
        # Plot-specific parameters
        st.markdown("#### Plot Parameters")
        
        plot_params = {}
        
        if plot_type == "Bin Scatter":
            plot_params['bin_size'] = st.slider(
                "Number of bins",
                min_value=5,
                max_value=50,
                value=10,
                help="Number of bins to group data into"
            )
            plot_params['bin_plot_type'] = st.selectbox(
                "Bin plot style",
                ["scatter", "bar"],
                help="Display bins as scatter points or bars"
            )
        elif plot_type == "Line":
            plot_params['line_smoothing'] = st.slider(
                "Line smoothing",
                min_value=0.0,
                max_value=1.3,
                value=0.0,
                step=0.1,
                help="Apply smoothing to line plots"
            )
        elif plot_type == "Scatter":
            plot_params['marker_size'] = st.slider(
                "Marker size",
                min_value=2,
                max_value=20,
                value=6,
                help="Size of scatter plot markers"
            )
        elif plot_type == "Bar":
            plot_params['bar_width'] = st.slider(
                "Bar width",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Width of bars relative to available space"
            )
        
        st.markdown("---")
        
        # Transform settings
        st.markdown("### Data Transforms")
        
        transform_type = st.selectbox(
            "Apply transform",
            ["none", "lowpass", "highpass", "bandpass", "savgol", "moving_average", "exponential_smoothing", "detrend"],
            help="Apply signal processing transforms to data"
        )
        
        transform_params = {}
        
        if transform_type == "lowpass":
            transform_params['cutoff'] = st.slider("Cutoff frequency", 0.01, 0.5, 0.1, 0.01)
            transform_params['order'] = st.slider("Filter order", 1, 10, 5)
        elif transform_type == "highpass":
            transform_params['cutoff'] = st.slider("Cutoff frequency", 0.01, 0.5, 0.1, 0.01)
            transform_params['order'] = st.slider("Filter order", 1, 10, 5)
        elif transform_type == "bandpass":
            transform_params['low'] = st.slider("Low cutoff", 0.01, 0.4, 0.05, 0.01)
            transform_params['high'] = st.slider("High cutoff", 0.1, 0.5, 0.2, 0.01)
            transform_params['order'] = st.slider("Filter order", 1, 10, 5)
        elif transform_type == "savgol":
            transform_params['window_length'] = st.slider("Window length", 5, 51, 11, 2)
            transform_params['polyorder'] = st.slider("Polynomial order", 1, 6, 3)
        elif transform_type == "moving_average":
            transform_params['window'] = st.slider("Window size", 3, 100, 10)
        elif transform_type == "exponential_smoothing":
            transform_params['alpha'] = st.slider("Smoothing factor", 0.01, 1.0, 0.3, 0.01)
        elif transform_type == "detrend":
            transform_params['method'] = st.selectbox("Detrend method", ["linear", "constant"])
        
        st.markdown("---")
        
        # Data filtering
        st.markdown("### Data Filtering")
        
        # Convert time column to datetime if needed
        if time_col:
            if df[time_col].dtype != 'datetime64[ns]':
                df[time_col] = pd.to_datetime(df[time_col])
            
            min_date = df[time_col].min()
            max_date = df[time_col].max()
            
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        st.markdown("---")
        
        # Reset button
        if st.button("Reset Dashboard", type="secondary", use_container_width=True):
            st.session_state.file_uploaded = False
            st.session_state.data = None
            st.rerun()
    
    # Main area for plots
    st.title("Time Series Visualization")
    
    # Apply date filtering
    if time_col and len(date_range) == 2:
        mask = (df[time_col] >= pd.to_datetime(date_range[0])) & \
               (df[time_col] <= pd.to_datetime(date_range[1]))
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    # Create analysis and plot
    try:
        # Validate inputs based on analysis mode
        analysis_params = {
            'time_col': time_col,
            'value_cols': value_cols,
            'x_col': x_col,
            'y_col': y_col,
            'lag': lag if analysis_mode == "Time Series Analysis" else None,
            'shift': shift if analysis_mode == "Correlation Analysis" else None
        }
        
        # Clean up params based on mode
        if analysis_mode == "Univariate Analysis":
            analysis_params = {'time_col': time_col, 'value_cols': value_cols}
        elif analysis_mode == "Time Series Analysis":
            analysis_params = {'time_col': time_col, 'value_col': y_col, 'lag': lag}
        elif analysis_mode == "Correlation Analysis":
            analysis_params = {'time_col': time_col, 'x_col': x_col, 'y_col': y_col, 'shift': shift}
        
        # Validate parameters
        is_valid, validation_message = processor.validate_analysis_params(analysis_mode, **analysis_params)
        
        if not is_valid:
            st.warning(f"Invalid parameters: {validation_message}")
        else:
            # Create analysis object
            analysis = processor.create_analysis(analysis_mode, **analysis_params)
            
            # Create plot
            fig = analysis.create_plot(
                plot_type=plot_type,
                date_range=date_range if 'date_range' in locals() else None,
                transform_type=transform_type,
                transform_params=transform_params,
                plot_params=plot_params,
                show_markers=show_markers
            )
            
            # Update layout
            fig.update_layout(height=plot_height)
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info tabs
            tab1, tab2, tab3 = st.tabs(["Summary Metrics", "Data Preview", "Export"])
            
            with tab1:
                st.markdown("### Summary Metrics")
                metrics = analysis.calculate_metrics()
                
                if analysis_mode == "Univariate Analysis":
                    for col, stats in metrics.items():
                        st.markdown(f"**{col}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean", f"{stats['mean']:.3f}")
                            st.metric("Std Dev", f"{stats['std']:.3f}")
                        with col2:
                            st.metric("Min", f"{stats['min']:.3f}")
                            st.metric("Max", f"{stats['max']:.3f}")
                        with col3:
                            st.metric("Median", f"{stats['median']:.3f}")
                            st.metric("Count", stats['count'])
                        st.markdown("---")
                        
                elif analysis_mode == "Time Series Analysis":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Lag", metrics['lag'])
                        st.metric("Autocorrelation at Lag", f"{metrics['autocorr_at_lag']:.3f}")
                    with col2:
                        st.metric("Pearson R", f"{metrics['pearson_r']:.3f}")
                        st.metric("R²", f"{metrics['r_squared']:.3f}")
                        
                elif analysis_mode == "Correlation Analysis":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pearson R", f"{metrics['pearson_r']:.3f}")
                        st.metric("Spearman R", f"{metrics['spearman_r']:.3f}")
                        st.metric("R²", f"{metrics['r_squared']:.3f}")
                    with col2:
                        st.metric("Applied Shift", metrics['shift_applied'])
                        st.metric("Optimal Lag", metrics['optimal_lag'])
                        st.metric("Max Cross-Corr", f"{metrics['max_cross_correlation']:.3f}")
            
            with tab2:
                st.markdown("### Data Preview")
                preview_df = analysis.prepare_data(date_range if 'date_range' in locals() else None)
                relevant_cols = [time_col] + ([col for col in value_cols if col] if value_cols else [])
                if relevant_cols:
                    st.dataframe(preview_df[relevant_cols].head(100), use_container_width=True)
            
            with tab3:
                st.markdown("### Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export filtered data as CSV
                    export_df = analysis.prepare_data(date_range if 'date_range' in locals() else None)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download analysis data as CSV",
                        data=csv,
                        file_name=f"{analysis_mode.lower().replace(' ', '_')}_{st.session_state.filename}",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export plot as HTML
                    html = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="Download plot as HTML",
                        data=html,
                        file_name=f"{analysis_mode.lower().replace(' ', '_')}_plot.html",
                        mime="text/html"
                    )
                    
    except Exception as e:
        st.error(f"Error creating analysis: {str(e)}")
        st.exception(e)