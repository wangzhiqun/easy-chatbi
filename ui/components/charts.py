"""
Chart components for ChatBI platform.
Handles data visualization using Plotly and Streamlit charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from typing import List, Dict, Any, Optional
import numpy as np


def render_chart_from_config(chart_config: Dict[str, Any], data: List[Dict[str, Any]]):
    """
    Render a chart based on configuration and data.

    Args:
        chart_config: Chart configuration dictionary
        data: Data to visualize
    """
    if not data or not chart_config:
        st.warning("No data or chart configuration provided.")
        return

    try:
        df = pd.DataFrame(data)
        chart_type = chart_config.get('chart_type', 'bar').lower()

        # Route to appropriate chart function
        if chart_type == 'bar':
            render_bar_chart(df, chart_config)
        elif chart_type == 'line':
            render_line_chart(df, chart_config)
        elif chart_type == 'pie':
            render_pie_chart(df, chart_config)
        elif chart_type == 'scatter':
            render_scatter_chart(df, chart_config)
        elif chart_type == 'area':
            render_area_chart(df, chart_config)
        elif chart_type == 'histogram':
            render_histogram(df, chart_config)
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            render_bar_chart(df, chart_config)  # Fallback to bar chart

    except Exception as e:
        st.error(f"Failed to render chart: {e}")
        # Show data table as fallback
        st.subheader("Data Preview (Chart failed to render)")
        st.dataframe(df.head(10))


def render_bar_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """Render bar chart."""
    x_axis = config.get('x_axis')
    y_axis = config.get('y_axis')
    title = config.get('title', 'Bar Chart')
    color_column = config.get('color_column')

    if not x_axis or not y_axis:
        st.error("X-axis and Y-axis must be specified for bar chart")
        return

    # Handle multiple y-axes
    if isinstance(y_axis, list):
        y_columns = y_axis
    else:
        y_columns = [y_axis]

    # Create bar chart
    fig = go.Figure()

    for y_col in y_columns:
        if y_col in df.columns:
            fig.add_trace(go.Bar(
                x=df[x_axis],
                y=df[y_col],
                name=y_col,
                text=df[y_col],
                textposition='auto',
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis.replace('_', ' ').title(),
        yaxis_title=', '.join(y_columns).replace('_', ' ').title(),
        hovermode='x unified',
        showlegend=len(y_columns) > 1
    )

    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """Render line chart."""
    x_axis = config.get('x_axis')
    y_axis = config.get('y_axis')
    title = config.get('title', 'Line Chart')
    color_column = config.get('color_column')

    if not x_axis or not y_axis:
        st.error("X-axis and Y-axis must be specified for line chart")
        return

    # Handle multiple y-axes
    if isinstance(y_axis, list):
        y_columns = y_axis
    else:
        y_columns = [y_axis]

    fig = go.Figure()

    for y_col in y_columns:
        if y_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_axis],
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(width=3),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis.replace('_', ' ').title(),
        yaxis_title=', '.join(y_columns).replace('_', ' ').title(),
        hovermode='x unified',
        showlegend=len(y_columns) > 1
    )

    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """Render pie chart."""
    x_axis = config.get('x_axis')  # Labels
    y_axis = config.get('y_axis')  # Values
    title = config.get('title', 'Pie Chart')

    if not x_axis or not y_axis:
        st.error("Both label and value columns must be specified for pie chart")
        return

    # Limit to top categories to avoid cluttered pie chart
    df_sorted = df.nlargest(10, y_axis)

    fig = go.Figure(data=[go.Pie(
        labels=df_sorted[x_axis],
        values=df_sorted[y_axis],
        hole=0.3,  # Donut chart
        textinfo='label+percent',
        textposition='auto'
    )])

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    if len(df) > 10:
        st.info(f"Showing top 10 of {len(df)} categories")


def render_scatter_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """Render scatter plot."""
    x_axis = config.get('x_axis')
    y_axis = config.get('y_axis')
    title = config.get('title', 'Scatter Plot')
    color_column = config.get('color_column')

    if not x_axis or not y_axis:
        st.error("X-axis and Y-axis must be specified for scatter plot")
        return

    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=color_column if color_column and color_column in df.columns else None,
        title=title,
        hover_data=df.columns.tolist()
    )

    fig.update_traces(marker=dict(size=8))
    fig.update_layout(hovermode='closest')

    st.plotly_chart(fig, use_container_width=True)

    # Show correlation if both axes are numeric
    if df[x_axis].dtype in ['int64', 'float64'] and df[y_axis].dtype in ['int64', 'float64']:
        correlation = df[x_axis].corr(df[y_axis])
        st.info(f"Correlation coefficient: {correlation:.3f}")


def render_area_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """Render area chart."""
    x_axis = config.get('x_axis')
    y_axis = config.get('y_axis')
    title = config.get('title', 'Area Chart')

    if not x_axis or not y_axis:
        st.error("X-axis and Y-axis must be specified for area chart")
        return

    # Handle multiple y-axes
    if isinstance(y_axis, list):
        y_columns = y_axis
    else:
        y_columns = [y_axis]

    fig = go.Figure()

    for i, y_col in enumerate(y_columns):
        if y_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_axis],
                y=df[y_col],
                fill='tonexty' if i > 0 else 'tozeroy',
                mode='lines',
                name=y_col,
                line=dict(width=0.5),
                stackgroup='one' if len(y_columns) > 1 else None
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis.replace('_', ' ').title(),
        yaxis_title=', '.join(y_columns).replace('_', ' ').title(),
        hovermode='x unified',
        showlegend=len(y_columns) > 1
    )

    st.plotly_chart(fig, use_container_width=True)


def render_histogram(df: pd.DataFrame, config: Dict[str, Any]):
    """Render histogram."""
    x_axis = config.get('x_axis')
    title = config.get('title', 'Histogram')

    if not x_axis:
        st.error("X-axis must be specified for histogram")
        return

    # Only works with numeric data
    if df[x_axis].dtype not in ['int64', 'float64']:
        st.warning(f"Histogram requires numeric data. {x_axis} is not numeric.")
        return

    fig = px.histogram(
        df,
        x=x_axis,
        title=title,
        nbins=min(30, len(df.dropna()) // 2)  # Dynamic bin size
    )

    fig.update_layout(
        xaxis_title=x_axis.replace('_', ' ').title(),
        yaxis_title='Frequency'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show statistics
    stats = df[x_axis].describe()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{stats['mean']:.2f}")
    with col2:
        st.metric("Median", f"{stats['50%']:.2f}")
    with col3:
        st.metric("Std Dev", f"{stats['std']:.2f}")
    with col4:
        st.metric("Range", f"{stats['max'] - stats['min']:.2f}")


def create_metric_chart(title: str, value: float, previous_value: Optional[float] = None):
    """Create a metric chart with trend indicator."""

    col1, col2 = st.columns([2, 1])

    with col1:
        delta = None
        if previous_value is not None:
            delta = value - previous_value

        st.metric(
            label=title,
            value=f"{value:,.0f}" if isinstance(value, (int, float)) else str(value),
            delta=f"{delta:,.0f}" if delta is not None else None
        )

    with col2:
        if previous_value is not None:
            # Create mini trend chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[previous_value, value],
                mode='lines+markers',
                line=dict(color='green' if value >= previous_value else 'red', width=3),
                marker=dict(size=8),
                showlegend=False
            ))

            fig.update_layout(
                height=100,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)


def create_usage_chart(data: List[Dict[str, Any]], title: str = "Usage Over Time"):
    """Create usage/activity chart."""
    if not data:
        st.info("No usage data available")
        return

    df = pd.DataFrame(data)

    fig = px.line(
        df,
        x='date' if 'date' in df.columns else df.columns[0],
        y='count' if 'count' in df.columns else df.columns[1],
        title=title,
        markers=True
    )

    fig.update_layout(
        hovermode='x unified',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str = "Heatmap"):
    """Render heatmap visualization."""

    # Pivot data for heatmap
    try:
        pivot_df = df.pivot(index=y_col, columns=x_col, values=value_col)

        fig = px.imshow(
            pivot_df,
            title=title,
            aspect='auto',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not create heatmap: {e}")


def render_box_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Box Plot"):
    """Render box plot for statistical analysis."""

    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=title
    )

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )

    st.plotly_chart(fig, use_container_width=True)


def render_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix"):
    """Render correlation matrix for numeric columns."""

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        st.warning("No numeric columns found for correlation analysis")
        return

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    fig = px.imshow(
        corr_matrix,
        title=title,
        aspect='auto',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        width=600,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def render_multi_chart_dashboard(data_configs: List[Dict[str, Any]]):
    """Render multiple charts in a dashboard layout."""

    if not data_configs:
        st.info("No charts to display")
        return

    # Create columns based on number of charts
    num_charts = len(data_configs)

    if num_charts == 1:
        render_chart_from_config(data_configs[0]['config'], data_configs[0]['data'])
    elif num_charts == 2:
        col1, col2 = st.columns(2)
        with col1:
            render_chart_from_config(data_configs[0]['config'], data_configs[0]['data'])
        with col2:
            render_chart_from_config(data_configs[1]['config'], data_configs[1]['data'])
    else:
        # For more charts, use rows and columns
        for i in range(0, num_charts, 2):
            col1, col2 = st.columns(2)

            with col1:
                if i < num_charts:
                    render_chart_from_config(data_configs[i]['config'], data_configs[i]['data'])

            with col2:
                if i + 1 < num_charts:
                    render_chart_from_config(data_configs[i + 1]['config'], data_configs[i + 1]['data'])


def get_chart_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get chart type suggestions based on data characteristics."""
    suggestions = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Bar chart for categorical vs numeric
    if categorical_cols and numeric_cols:
        suggestions.append({
            'type': 'bar',
            'title': f'{categorical_cols[0]} vs {numeric_cols[0]}',
            'x_axis': categorical_cols[0],
            'y_axis': numeric_cols[0],
            'description': 'Compare values across categories'
        })

    # Line chart for time series
    if datetime_cols and numeric_cols:
        suggestions.append({
            'type': 'line',
            'title': f'{numeric_cols[0]} over time',
            'x_axis': datetime_cols[0],
            'y_axis': numeric_cols[0],
            'description': 'Show trends over time'
        })

    # Scatter plot for numeric relationships
    if len(numeric_cols) >= 2:
        suggestions.append({
            'type': 'scatter',
            'title': f'{numeric_cols[0]} vs {numeric_cols[1]}',
            'x_axis': numeric_cols[0],
            'y_axis': numeric_cols[1],
            'description': 'Explore relationships between variables'
        })

    # Pie chart for categorical distribution
    if categorical_cols and numeric_cols:
        suggestions.append({
            'type': 'pie',
            'title': f'Distribution of {categorical_cols[0]}',
            'x_axis': categorical_cols[0],
            'y_axis': numeric_cols[0],
            'description': 'Show composition and proportions'
        })

    return suggestions


def export_chart_config(chart_config: Dict[str, Any]) -> str:
    """Export chart configuration as JSON string."""
    import json
    return json.dumps(chart_config, indent=2)


def import_chart_config(config_json: str) -> Dict[str, Any]:
    """Import chart configuration from JSON string."""
    import json
    try:
        return json.loads(config_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON configuration: {e}")


def render_chart_controls(chart_config: Dict[str, Any], data: List[Dict[str, Any]]):
    """Render interactive controls for chart customization."""

    with st.expander("🎨 Chart Controls", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Chart type selector
            chart_types = ['bar', 'line', 'pie', 'scatter', 'area', 'histogram']
            current_type = chart_config.get('chart_type', 'bar')

            new_type = st.selectbox(
                "Chart Type",
                chart_types,
                index=chart_types.index(current_type) if current_type in chart_types else 0,
                key="chart_type_control"
            )

            # Title input
            new_title = st.text_input(
                "Chart Title",
                value=chart_config.get('title', ''),
                key="chart_title_control"
            )

        with col2:
            if data:
                columns = list(data[0].keys())

                # X-axis selector
                current_x = chart_config.get('x_axis', columns[0] if columns else '')
                new_x = st.selectbox(
                    "X-Axis",
                    columns,
                    index=columns.index(current_x) if current_x in columns else 0,
                    key="chart_x_control"
                )

                # Y-axis selector
                current_y = chart_config.get('y_axis',
                                             columns[1] if len(columns) > 1 else columns[0] if columns else '')
                new_y = st.selectbox(
                    "Y-Axis",
                    columns,
                    index=columns.index(current_y) if current_y in columns else 1 if len(columns) > 1 else 0,
                    key="chart_y_control"
                )

        # Update chart if controls changed
        if st.button("🔄 Update Chart", key="update_chart_button"):
            updated_config = {
                'chart_type': new_type,
                'title': new_title,
                'x_axis': new_x,
                'y_axis': new_y
            }

            try:
                render_chart_from_config(updated_config, data)
                st.success("Chart updated!")
            except Exception as e:
                st.error(f"Failed to update chart: {e}")


def create_summary_charts(df: pd.DataFrame):
    """Create a set of summary charts for data overview."""

    st.subheader("📊 Data Summary")

    # Data overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{100 - null_percentage:.1f}%")

    with col4:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)

    # Quick visualizations
    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        st.markdown("### 📈 Numeric Data Distribution")

        # Create histogram for first numeric column
        first_numeric = numeric_df.columns[0]
        fig = px.histogram(df, x=first_numeric, title=f"Distribution of {first_numeric}")
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap if multiple numeric columns
        if len(numeric_df.columns) > 1:
            st.markdown("### 🔥 Correlation Heatmap")
            render_correlation_matrix(numeric_df)