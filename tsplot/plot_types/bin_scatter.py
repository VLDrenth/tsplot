import plotly.graph_objects as go
import numpy as np
import pandas as pd

def _create_scatter_plot(grouped, x_col, y_col, color_col=None):
    """Helper function to create scatter plot."""
    marker_kwargs = {}
    if color_col:
        marker_kwargs = {
            'color': grouped[color_col],
            'colorscale': 'Viridis',
            'colorbar': dict(title=color_col),
            'showscale': True
        }

    return go.Figure(go.Scatter(
        x=grouped['x_center'],
        y=grouped[y_col],
        mode='markers',
        marker=marker_kwargs,
        name='Bin Means'
    ))

def _create_bar_plot(grouped, x_col, y_col, color_col=None):
    """Helper function to create bar plot."""
    if color_col:
        return go.Figure(go.Bar(
            x=grouped['x_center'],
            y=grouped[y_col],
            marker=dict(
                color=grouped[color_col],
                colorscale='Viridis',
                colorbar=dict(title=color_col),
                showscale=True
            ),
            name='Bin Means'
        ))
    else:
        return go.Figure(go.Bar(
            x=grouped['x_center'],
            y=grouped[y_col],
            name='Bin Means'
        ))

def bin_scatter(df, x_col, y_col, bin_size=10, color_col=None, plot_type='scatter'):
    """
    Generate a binned plot using mean of y_col in each x_col bin.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x_col : str
        Column to bin along x-axis.
    y_col : str
        Column to average and plot along y-axis.
    bin_size : int, default=10
        Number of bins along x-axis.
    color_col : str, optional
        Column to compute average color per bin.
    plot_type : str, default='scatter'
        Type of plot: 'scatter' for scatter plot or 'bar' for bar plot.

    Returns
    -------
    plotly.graph_objects.Figure
        A binned plot (scatter or bar).
    """
    # Handle datetime columns differently
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        # Convert to numeric timestamps for binning
        min_x = df[x_col].min().value  # nanoseconds since epoch
        max_x = df[x_col].max().value
        bins = np.linspace(min_x, max_x, bin_size + 1)
        # Convert back to datetime for pd.cut
        bins = pd.to_datetime(bins)
    else:
        # Compute bin edges using number of bins for numeric data
        min_x, max_x = df[x_col].min(), df[x_col].max()
        bins = np.linspace(min_x, max_x, bin_size + 1)

    # Assign bin labels
    df_binned = df.copy()
    df_binned['x_bin'] = pd.cut(df_binned[x_col], bins=bins, include_lowest=True)

    # Group and aggregate
    agg_dict = {y_col: 'mean'}
    if color_col:
        agg_dict[color_col] = 'mean'

    grouped = df_binned.groupby('x_bin').agg(agg_dict).reset_index()

    # Use bin center as x
    grouped['x_center'] = grouped['x_bin'].apply(lambda x: x.mid)

    # Create figure based on plot type using helper functions
    if plot_type == 'bar':
        fig = _create_bar_plot(grouped, x_col, y_col, color_col)
    else:  # scatter plot
        fig = _create_scatter_plot(grouped, x_col, y_col, color_col)

    plot_title = 'Binned Bar Plot' if plot_type == 'bar' else 'Binned Scatter Plot'
    fig.update_layout(
        title=plot_title,
        xaxis_title=f'{x_col} (with {bin_size} bins)',
        yaxis_title=f'Mean of {y_col}'
    )

    return fig

if __name__ == "__main__":
    # Generate simulated data with a relationship
    np.random.seed(42)  # For reproducible results
    n_points = 1000
    
    # Create x data
    x = np.random.uniform(0, 100, n_points)
    
    # Create y data with a quadratic relationship plus noise
    y = 0.02 * x**2 - x + 50 + np.random.normal(0, 8, n_points)
    
    # Create color data that varies with x
    color = 0.5 * x + np.random.normal(0, 5, n_points)
    
    data = {
        'x': x,
        'y': y,
        'color': color
    }
    df = pd.DataFrame(data)
    
    print("Generating binned scatter plot...")
    print(f"Data shape: {df.shape}")
    print(f"X range: {df['x'].min():.2f} to {df['x'].max():.2f}")
    print(f"Y range: {df['y'].min():.2f} to {df['y'].max():.2f}")
    
    # Test scatter plot
    print("\n--- Testing scatter plot with 15 bins ---")
    fig_scatter = bin_scatter(df, x_col='x', y_col='y', bin_size=15, color_col='color', plot_type='scatter')
    print(f"Scatter plot - Number of data points after binning: {len(fig_scatter.data[0]['x'])}")
    
    # Test bar plot
    print("\n--- Testing bar plot with 15 bins ---")
    fig_bar = bin_scatter(df, x_col='x', y_col='y', bin_size=15, color_col='color', plot_type='bar')
    print(f"Bar plot - Number of data points after binning: {len(fig_bar.data[0]['x'])}")
    
    # Show both plots
    fig_scatter.show()
    fig_bar.show()