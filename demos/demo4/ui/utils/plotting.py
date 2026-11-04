# ui/utils/plotting.py
from typing import Optional, Union, Any
import pandas as pd
import plotly.graph_objects as go


def plotly_layout(fig: go.Figure, title: str = "", height: int = 350) -> go.Figure:
    """Enforces a standardized, minimalistic layout on Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, family="sans-serif", color="#0D1B2A"),
            x=0,
            xanchor="left",
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#EEF2F7",
            linecolor="#D1D5DB",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EEF2F7",
            linecolor="#D1D5DB",
        ),
    )
    return fig


def build_ts_figure(
    df_or_title: Union[pd.DataFrame, str] = "",
    title: str = "Daily Precipitation Timeseries",
    height: int = 400,
    show_cumulative: bool = False,
    onset_date: Optional[str] = None,
    threshold: Optional[float] = None,
) -> go.Figure:
    """
    Builds a styled Plotly figure for daily precipitation time series.
    Supports receiving either a DataFrame or a title string as the first argument.
    """
    fig = go.Figure()

    # Handle string passed as first positional argument (e.g., build_ts_figure(title, height=...))
    if isinstance(df_or_title, str):
        chart_title = df_or_title if df_or_title else title
        fig = plotly_layout(fig, title=chart_title, height=height)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Precipitation (mm)")
        return fig

    df = df_or_title
    date_col = "time" if "time" in df.columns else "date" if "date" in df.columns else df.columns[0]
    rain_col = "rain" if "rain" in df.columns else "precip" if "precip" in df.columns else df.columns[1]

    fig = add_rain_trace(fig, df[date_col], df[rain_col])

    if show_cumulative and rain_col in df.columns:
        cum_rain = df[rain_col].fillna(0).cumsum()
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=cum_rain,
                name="Cumulative (mm)",
                mode="lines",
                line=dict(color="#10B981", width=2, dash="dash"),
                hovertemplate="%{x|%b %d, %Y}<br>Cumulative: %{y:.1f} mm<extra></extra>",
            )
        )

    if threshold is not None:
        fig = add_threshold_line(fig, threshold=threshold, x_series=df[date_col])

    if onset_date:
        fig = add_onset_markers(fig, onset_date=onset_date)

    fig = plotly_layout(fig, title=title, height=height)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Precipitation (mm)")

    return fig


def add_rain_trace(
    fig: go.Figure,
    x: Any = None,
    y: Any = None,
    label: str = "Daily Rain (mm)",
    color: str = "#2563EB",
    opacity: float = 0.85,
    markers: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """Adds a precipitation line or bar trace to an existing Plotly figure."""
    # Handle passing DataFrame as first argument or x/y series directly
    if isinstance(x, pd.DataFrame):
        df = x
        x_col = "time" if "time" in df.columns else "date" if "date" in df.columns else df.columns[0]
        y_col = "rain" if "rain" in df.columns else "precip" if "precip" in df.columns else df.columns[1]
        x_vals, y_vals = df[x_col], df[y_col]
    else:
        x_vals, y_vals = x, y

    mode = "lines+markers" if markers else "lines"
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            name=label,
            mode=mode,
            line=dict(color=color, width=1.5),
            marker=dict(size=4) if markers else None,
            hovertemplate="%{x|%b %d, %Y}<br>Rain: %{y:.1f} mm<extra></extra>",
        )
    )
    return fig


def add_onset_markers(
    fig: go.Figure,
    df_or_onset: Any = None,
    key: Optional[str] = None,
    station_or_label: Optional[str] = None,
    onset_params: Optional[dict] = None,
    onset_date: Optional[Union[str, pd.Timestamp]] = None,
    false_start_date: Optional[Union[str, pd.Timestamp]] = None,
    cessation_date: Optional[Union[str, pd.Timestamp]] = None,
) -> go.Figure:
    """Adds vertical annotations for Onset, False Start, or Cessation dates."""
    date_val = onset_date
    if date_val is None and isinstance(df_or_onset, (str, pd.Timestamp)):
        date_val = df_or_onset

    if date_val:
        fig.add_vline(
            x=date_val,
            line_width=2,
            line_dash="dash",
            line_color="#10B981",
            annotation_text=f"Onset ({key or ''})".strip(),
            annotation_position="top left",
            annotation_font=dict(color="#10B981", size=11, family="sans-serif"),
        )

    if false_start_date:
        fig.add_vline(
            x=false_start_date,
            line_width=2,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="False Start",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11, family="sans-serif"),
        )

    if cessation_date:
        fig.add_vline(
            x=cessation_date,
            line_width=2,
            line_dash="dash",
            line_color="#6366F1",
            annotation_text="Cessation",
            annotation_position="bottom left",
            annotation_font=dict(color="#6366F1", size=11, family="sans-serif"),
        )

    return fig


def add_threshold_line(
    fig: go.Figure,
    threshold: float,
    x_series: Any = None,
    label: str = "Wet Day Threshold",
    color: str = "#F59E0B",
    dash: str = "dash",
) -> go.Figure:
    """Adds a horizontal reference threshold line across a Plotly figure."""
    fig.add_hline(
        y=threshold,
        line_width=1.5,
        line_dash=dash,
        line_color=color,
        annotation_text=f"{label} ({threshold:.1f} mm)",
        annotation_position="top right",
        annotation_font=dict(color=color, size=10, family="sans-serif"),
    )
    return fig