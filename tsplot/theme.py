import plotly.io as pio


def register_tsplot_clean_theme(set_as_default: bool = True) -> None:
    """
    Register a light-mode Plotly template called 'tsplot_clean'.
    
    The aesthetic targets a clean, modern analytical look:
    - Fonts: Inter → sans-serif fallback
    - Colours: electric blue plus four accent hues
    - Layout: white canvas, subtle gridlines, generous margins
    - UX touches: horizontal legend, smooth transitions, pan as default dragmode
    
    Parameters
    ----------
    set_as_default : bool, optional
        If True (default) set the newly-registered template as
        pio.templates.default so every subsequent figure uses it automatically.
    
    Examples
    --------
    >>> import plotly.express as px
    >>> from tsplot.theme import register_tsplot_clean_theme
    >>> register_tsplot_clean_theme()
    >>> fig = px.line(df, x="date", y="price", title="Time Series Analysis")
    >>> fig.show()
    """
    pio.templates["tsplot_clean"] = dict(
        layout=dict(
            # Typography
            font=dict(family="Inter, sans-serif", size=14, color="#1b1b1b"),
            title=dict(font=dict(size=20, color="#1b1b1b", family="Inter, sans-serif")),
            
            # Canvas
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=60, r=40, t=60, b=60),
            
            # Colour palette
            colorway=["#0057FF", "#00B8A9", "#F6416C", "#FFDE7D", "#785EF0"],
            
            # Axes
            xaxis=dict(
                showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
                zeroline=False,
                showline=True, linecolor="#D0D0D0", linewidth=1,
                ticks="outside", tickcolor="#D0D0D0", ticklen=5,
                title=dict(font=dict(size=16))
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
                zeroline=False,
                showline=True, linecolor="#D0D0D0", linewidth=1,
                ticks="outside", tickcolor="#D0D0D0", ticklen=5,
                title=dict(font=dict(size=16))
            ),
            
            # Legend
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                orientation="h",
                x=0, y=1.08,
                font=dict(size=12, family="Inter, sans-serif")
            ),
            
            # Hover & interactivity
            hoverlabel=dict(
                font=dict(family="Inter, sans-serif", size=13),
                bgcolor="white",
                bordercolor="#C0C0C0"
            ),
            dragmode="pan",
            modebar=dict(
                bgcolor="rgba(0,0,0,0)",
                color="#666",
                activecolor="#0057FF"
            ),
            transition=dict(duration=300)
        )
    )
    
    if set_as_default:
        pio.templates.default = "tsplot_clean"


def get_available_themes():
    """Get list of available theme names."""
    return ["tsplot_clean"]


def apply_theme(theme_name: str = "tsplot_clean"):
    """Apply a specific theme."""
    if theme_name == "tsplot_clean":
        register_tsplot_clean_theme(set_as_default=True)
    else:
        raise ValueError(f"Unknown theme: {theme_name}")