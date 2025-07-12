import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

st.set_page_config(page_title="Time Series Dashboard", layout="wide")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

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
            st.session_state.file_uploaded = True
            st.session_state.filename = uploaded_file.name
            st.rerun()
            
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Main dashboard
else:
    df = st.session_state.data
    
    # Sidebar for settings
    with st.sidebar:
        st.title("Dashboard Settings")
        
        # File info
        st.markdown("### Data Info")
        st.info(f"File: {st.session_state.filename}")
        st.info(f"Rows: {len(df):,}")
        st.info(f"Columns: {len(df.columns)}")
        
        st.markdown("---")
        
        # Column selection
        st.markdown("### Column Selection")
        
        # Detect datetime columns
        datetime_cols = []
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    pass
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        
        # Time column selection
        time_col = st.selectbox(
            "Select time column",
            options=datetime_cols if datetime_cols else df.columns.tolist(),
            help="Select the column containing timestamps"
        )
        
        # Value columns selection
        value_cols = st.multiselect(
            "Select value columns to plot",
            options=numeric_cols if numeric_cols else df.columns.tolist(),
            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
            help="Select one or more numeric columns to visualize"
        )
        
        st.markdown("---")
        
        # Plot settings
        st.markdown("### Plot Settings")
        
        plot_type = st.selectbox(
            "Chart type",
            ["Line", "Scatter", "Bar", "Area"],
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
    
    # Create plot based on settings
    if value_cols and time_col:
        # Prepare data for plotting
        plot_df = filtered_df[[time_col] + value_cols].copy()
        plot_df = plot_df.sort_values(time_col)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces based on plot type
        for col in value_cols:
            if plot_type == "Line":
                fig.add_trace(go.Scatter(
                    x=plot_df[time_col],
                    y=plot_df[col],
                    mode='lines+markers' if show_markers else 'lines',
                    name=col
                ))
            elif plot_type == "Scatter":
                fig.add_trace(go.Scatter(
                    x=plot_df[time_col],
                    y=plot_df[col],
                    mode='markers',
                    name=col
                ))
            elif plot_type == "Bar":
                fig.add_trace(go.Bar(
                    x=plot_df[time_col],
                    y=plot_df[col],
                    name=col
                ))
            elif plot_type == "Area":
                fig.add_trace(go.Scatter(
                    x=plot_df[time_col],
                    y=plot_df[col],
                    mode='lines',
                    fill='tozeroy',
                    name=col
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{plot_type} Chart: {', '.join(value_cols)}",
            xaxis_title=time_col,
            yaxis_title="Values",
            height=plot_height,
            hovermode='x unified',
            showlegend=True if len(value_cols) > 1 else False
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional info tabs
        tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Data Preview", "Export"])
        
        with tab1:
            st.markdown("### Summary Statistics")
            summary_df = filtered_df[value_cols].describe()
            st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.markdown("### Data Preview")
            st.dataframe(
                filtered_df[[time_col] + value_cols].head(100),
                use_container_width=True
            )
        
        with tab3:
            st.markdown("### Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export filtered data as CSV
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name=f"filtered_{st.session_state.filename}",
                    mime="text/csv"
                )
            
            with col2:
                # Export plot as HTML
                html = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="Download plot as HTML",
                    data=html,
                    file_name="plot.html",
                    mime="text/html"
                )
    else:
        st.warning("Please select time column and at least one value column from the sidebar.")