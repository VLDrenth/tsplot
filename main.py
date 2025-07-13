import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

from tsplot.plot_types.bin_scatter import bin_scatter
from tsplot.transforms import apply_transform, resample_timeseries, TransformPipeline, get_transform_info
from tsplot.data_processor import DataProcessor
from tsplot.theme import register_tsplot_clean_theme

st.set_page_config(page_title="Time Series Dashboard", layout="wide")

# Initialize theme
register_tsplot_clean_theme()

def render_transform_pipeline():
    """Render the transform pipeline UI."""
    st.markdown("### Transform Pipeline")
    
    pipeline = st.session_state.transform_pipeline
    transform_info = get_transform_info()
    
    # Transform pipeline header with controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Active Transforms: {len(pipeline.transforms)}/3**")
    
    with col2:
        if st.button("Clear All", disabled=pipeline.is_empty()):
            pipeline.clear()
            st.session_state.selected_transform_id = None
            st.rerun()
    
    with col3:
        show_add = not pipeline.is_full()
        if st.button("+ Add Transform", disabled=not show_add, type="primary" if show_add else "secondary"):
            st.session_state.show_add_transform = True
            st.rerun()
    
    # Add transform dialog
    if st.session_state.get('show_add_transform', False):
        with st.expander("Add New Transform", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_transform_type = st.selectbox(
                    "Transform Type",
                    options=list(transform_info.keys()),
                    format_func=lambda x: transform_info[x]["name"],
                    key="new_transform_type"
                )
            
            with col2:
                if st.button("Add", key="add_transform_btn"):
                    try:
                        default_params = {}
                        for param_name, param_info in transform_info[new_transform_type]["parameters"].items():
                            default_params[param_name] = param_info["default"]
                        
                        pipeline.add_transform(new_transform_type, default_params)
                        st.session_state.show_add_transform = False
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                
                if st.button("Cancel", key="cancel_add_transform"):
                    st.session_state.show_add_transform = False
                    st.rerun()
    
    # Display pipeline transforms
    if not pipeline.is_empty():
        st.markdown("#### Pipeline Steps")
        
        for i, transform in enumerate(pipeline.transforms):
            render_transform_card(transform, i, transform_info)
    else:
        st.info("No transforms in pipeline. Click '+ Add Transform' to get started.")

def render_transform_card(transform, position, transform_info):
    """Render a single transform card."""
    transform_config = transform_info.get(transform.transform_type, {})
    
    # Card container
    with st.container():
        # Header with position, name, and controls
        col1, col2, col3, col4, col5 = st.columns([0.5, 2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{position + 1}.**")
        
        with col2:
            enabled_icon = "✅" if transform.enabled else "⏸️"
            st.markdown(f"{enabled_icon} **{transform_config.get('name', transform.transform_type)}**")
        
        with col3:
            # Move up button
            if position > 0:
                if st.button("↑", key=f"up_{transform.id}", help="Move up"):
                    st.session_state.transform_pipeline.move_transform(transform.id, position - 1)
                    st.rerun()
        
        with col4:
            # Move down button
            if position < len(st.session_state.transform_pipeline.transforms) - 1:
                if st.button("↓", key=f"down_{transform.id}", help="Move down"):
                    st.session_state.transform_pipeline.move_transform(transform.id, position + 1)
                    st.rerun()
        
        with col5:
            # Remove button
            if st.button("🗑️", key=f"remove_{transform.id}", help="Remove transform"):
                st.session_state.transform_pipeline.remove_transform(transform.id)
                if st.session_state.selected_transform_id == transform.id:
                    st.session_state.selected_transform_id = None
                st.rerun()
        
        # Transform description
        if transform_config.get('description'):
            st.caption(transform_config['description'])
        
        # Toggle enabled/disabled
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(
                "Disable" if transform.enabled else "Enable",
                key=f"toggle_{transform.id}",
                type="secondary"
            ):
                st.session_state.transform_pipeline.toggle_transform(transform.id)
                st.rerun()
        
        with col2:
            # Configure button
            is_selected = st.session_state.selected_transform_id == transform.id
            if st.button(
                "Hide Config" if is_selected else "Configure",
                key=f"config_{transform.id}",
                type="primary" if not is_selected else "secondary"
            ):
                if is_selected:
                    st.session_state.selected_transform_id = None
                else:
                    st.session_state.selected_transform_id = transform.id
                st.rerun()
        
        # Parameter configuration (expanded if selected)
        if st.session_state.selected_transform_id == transform.id:
            render_transform_parameters(transform, transform_config)
        
        st.markdown("---")

def render_transform_parameters(transform, transform_config):
    """Render parameter controls for a transform."""
    st.markdown("##### Parameters")
    
    parameters = transform_config.get("parameters", {})
    updated_params = {}
    
    for param_name, param_info in parameters.items():
        current_value = transform.parameters.get(param_name, param_info["default"])
        
        if param_info["type"] == "float":
            updated_params[param_name] = st.slider(
                param_name.replace('_', ' ').title(),
                min_value=param_info["min"],
                max_value=param_info["max"],
                value=current_value,
                step=param_info.get("step", 0.01),
                key=f"param_{transform.id}_{param_name}"
            )
        elif param_info["type"] == "int":
            step = param_info.get("step", 1)
            if param_name == "window_length":
                # Ensure odd numbers for savgol filter
                min_val = param_info["min"]
                max_val = param_info["max"]
                value = current_value
                if value % 2 == 0:
                    value += 1
                
                updated_params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=value,
                    step=2,  # Force odd numbers
                    key=f"param_{transform.id}_{param_name}"
                )
            else:
                updated_params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=param_info["min"],
                    max_value=param_info["max"],
                    value=current_value,
                    step=step,
                    key=f"param_{transform.id}_{param_name}"
                )
        elif param_info["type"] == "select":
            updated_params[param_name] = st.selectbox(
                param_name.replace('_', ' ').title(),
                options=param_info["options"],
                index=param_info["options"].index(current_value) if current_value in param_info["options"] else 0,
                key=f"param_{transform.id}_{param_name}"
            )
    
    # Update parameters if they changed
    if updated_params != transform.parameters:
        st.session_state.transform_pipeline.update_transform_parameters(transform.id, updated_params)

if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'transform_pipeline' not in st.session_state:
    st.session_state.transform_pipeline = TransformPipeline()
if 'selected_transform_id' not in st.session_state:
    st.session_state.selected_transform_id = None
if 'enable_resampling' not in st.session_state:
    st.session_state.enable_resampling = False
if 'resample_frequency' not in st.session_state:
    st.session_state.resample_frequency = 'W'  # Default to Weekly
if 'resample_strategy' not in st.session_state:
    st.session_state.resample_strategy = 'mean'

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
        
        # Theme selector
        st.markdown("### Theme")
        theme_choice = st.selectbox(
            "Plot theme",
            ["tsplot_clean"],
            help="Choose the visual theme for plots"
        )
        
        # Apply theme if changed
        if theme_choice == "tsplot_clean":
            register_tsplot_clean_theme()
        
        st.markdown("---")
        
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
        
        # Transform pipeline
        render_transform_pipeline()
        
        st.markdown("---")
        
        # Resampling settings
        st.markdown("### Data Resampling")
        
        # Use session state to persist resampling settings
        enable_resampling = st.checkbox(
            "Enable resampling", 
            value=st.session_state.enable_resampling,
            key="resample_checkbox",
            help="Resample time series to different frequencies"
        )
        
        # Update session state when checkbox changes
        if enable_resampling != st.session_state.enable_resampling:
            st.session_state.enable_resampling = enable_resampling
        
        resample_params = {}
        if enable_resampling:
            col1, col2 = st.columns(2)
            with col1:
                frequency_options = ["D", "W", "M", "Q", "Y"]
                current_freq_index = frequency_options.index(st.session_state.resample_frequency) if st.session_state.resample_frequency in frequency_options else 1
                
                new_frequency = st.selectbox(
                    "Frequency",
                    frequency_options,
                    index=current_freq_index,
                    key="resample_frequency_select",
                    help="D=Daily, W=Weekly, M=Monthly, Q=Quarterly, Y=Yearly"
                )
                
                # Update session state when frequency changes
                if new_frequency != st.session_state.resample_frequency:
                    st.session_state.resample_frequency = new_frequency
                
                resample_params['frequency'] = st.session_state.resample_frequency
                
            with col2:
                strategy_options = ["mean", "last", "first", "sum", "max", "min", "count"]
                current_strategy_index = strategy_options.index(st.session_state.resample_strategy) if st.session_state.resample_strategy in strategy_options else 0
                
                new_strategy = st.selectbox(
                    "Strategy",
                    strategy_options,
                    index=current_strategy_index,
                    key="resample_strategy_select",
                    help="How to aggregate data within each period"
                )
                
                # Update session state when strategy changes
                if new_strategy != st.session_state.resample_strategy:
                    st.session_state.resample_strategy = new_strategy
                
                resample_params['strategy'] = st.session_state.resample_strategy
        
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
                transform_pipeline=st.session_state.transform_pipeline,
                plot_params=plot_params,
                show_markers=show_markers,
                resample_params=resample_params if st.session_state.enable_resampling else None
            )
            
            # Update layout
            fig.update_layout(height=plot_height)
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info tabs
            tab1, tab2, tab3 = st.tabs(["Summary Metrics", "Data Preview", "Export"])
            
            with tab1:
                st.markdown("### Summary Metrics")
                metrics = analysis.calculate_metrics(
                    transform_pipeline=st.session_state.transform_pipeline,
                    date_range=date_range if 'date_range' in locals() else None,
                    resample_params=resample_params if st.session_state.enable_resampling else None
                )
                
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
                preview_df = analysis.prepare_data(
                    date_range if 'date_range' in locals() else None,
                    resample_params if st.session_state.enable_resampling else None
                )
                relevant_cols = [time_col] + ([col for col in value_cols if col] if value_cols else [])
                if relevant_cols:
                    st.dataframe(preview_df[relevant_cols].head(100), use_container_width=True)
            
            with tab3:
                st.markdown("### Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export filtered data as CSV
                    export_df = analysis.prepare_data(
                        date_range if 'date_range' in locals() else None,
                        resample_params if st.session_state.enable_resampling else None
                    )
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