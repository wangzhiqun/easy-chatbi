import os
import sys
from typing import Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def render_chart_builder():
    st.subheader("📈 Interactive Chart Builder")

    data_source = st.radio(
        "Select Data Source",
        ["Upload File", "Query Result", "Sample Data"],
        horizontal=True
    )

    df = load_data_source(data_source)

    if df is not None:
        col1, col2 = st.columns([2, 1])

        with col2:
            chart_config = configure_chart(df)

        with col1:
            if chart_config:
                render_chart(chart_config, df)

        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(50))


def load_data_source(source_type: str) -> Optional[pd.DataFrame]:
    df = None

    if source_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json']
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)

                st.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif source_type == "Query Result":
        if 'query_result' in st.session_state and st.session_state.query_result:
            data = st.session_state.query_result.get('data')
            if data:
                df = pd.DataFrame(data)
                st.success(f"Using query result with {len(df)} rows")
        else:
            st.info("No query result available. Execute a query first.")

    elif source_type == "Sample Data":
        df = load_sample_data()
        st.success("Loaded sample data")

    return df


def load_sample_data() -> pd.DataFrame:
    import numpy as np

    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    data = {
        'Date': dates,
        'Revenue': np.random.randint(1000, 5000, 30),
        'Orders': np.random.randint(10, 100, 30),
        'Category': np.random.choice(['Electronics', 'Furniture', 'Stationery'], 30),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 30)
    }

    return pd.DataFrame(data)


def configure_chart(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    st.subheader("Chart Configuration")

    chart_type = st.selectbox(
        "Chart Type",
        ["Line", "Bar", "Scatter", "Pie", "Heatmap", "Box", "Histogram", "Area"]
    )

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    all_cols = df.columns.tolist()

    config = {'type': chart_type.lower()}

    if chart_type in ["Line", "Bar", "Scatter", "Area"]:
        config['x'] = st.selectbox("X Axis", all_cols)
        config['y'] = st.selectbox("Y Axis", numeric_cols if numeric_cols else all_cols)

        if categorical_cols:
            color_by = st.selectbox("Color By (optional)", ["None"] + categorical_cols)
            if color_by != "None":
                config['color'] = color_by

    elif chart_type == "Pie":
        config['values'] = st.selectbox("Values", numeric_cols)
        config['names'] = st.selectbox("Names", categorical_cols if categorical_cols else all_cols)

    elif chart_type == "Heatmap":
        config['x'] = st.selectbox("X Axis", all_cols)
        config['y'] = st.selectbox("Y Axis", all_cols)
        config['z'] = st.selectbox("Values", numeric_cols)

    elif chart_type in ["Box", "Histogram"]:
        config['column'] = st.selectbox("Column", numeric_cols)
        if chart_type == "Box" and categorical_cols:
            group_by = st.selectbox("Group By (optional)", ["None"] + categorical_cols)
            if group_by != "None":
                config['group'] = group_by

    with st.expander("Advanced Options"):
        config['title'] = st.text_input("Chart Title", f"{chart_type} Chart")
        config['height'] = st.slider("Height", 300, 800, 400)

        if chart_type != "Pie":
            config['show_legend'] = st.checkbox("Show Legend", True)
            config['show_grid'] = st.checkbox("Show Grid", True)

    return config


def render_chart(config: Dict[str, Any], df: pd.DataFrame):
    chart_type = config.get('type', 'bar')

    try:
        if chart_type == 'line':
            fig = create_line_chart(config, df)
        elif chart_type == 'bar':
            fig = create_bar_chart(config, df)
        elif chart_type == 'scatter':
            fig = create_scatter_chart(config, df)
        elif chart_type == 'pie':
            fig = create_pie_chart(config, df)
        elif chart_type == 'heatmap':
            fig = create_heatmap(config, df)
        elif chart_type == 'box':
            fig = create_box_plot(config, df)
        elif chart_type == 'histogram':
            fig = create_histogram(config, df)
        elif chart_type == 'area':
            fig = create_area_chart(config, df)
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return

        fig.update_layout(
            title=config.get('title', ''),
            height=config.get('height', 400),
            showlegend=config.get('show_legend', True),
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📷 Save as Image"):
                fig.write_image("chart.png")
                st.success("Chart saved as chart.png")

        with col2:
            if st.button("📄 Save as HTML"):
                fig.write_html("chart.html")
                st.success("Chart saved as chart.html")

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")


def create_line_chart(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    return px.line(
        df,
        x=config.get('x'),
        y=config.get('y'),
        color=config.get('color'),
        title=config.get('title', 'Line Chart')
    )


def create_bar_chart(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    return px.bar(
        df,
        x=config.get('x'),
        y=config.get('y'),
        color=config.get('color'),
        title=config.get('title', 'Bar Chart')
    )


def create_scatter_chart(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    return px.scatter(
        df,
        x=config.get('x'),
        y=config.get('y'),
        color=config.get('color'),
        title=config.get('title', 'Scatter Plot')
    )


def create_pie_chart(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    return px.pie(
        df,
        values=config.get('values'),
        names=config.get('names'),
        title=config.get('title', 'Pie Chart')
    )


def create_heatmap(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    pivot_df = df.pivot_table(
        index=config.get('y'),
        columns=config.get('x'),
        values=config.get('z'),
        aggfunc='mean'
    )

    return px.imshow(
        pivot_df,
        title=config.get('title', 'Heatmap'),
        color_continuous_scale='Viridis'
    )


def create_box_plot(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    if config.get('group'):
        return px.box(
            df,
            x=config.get('group'),
            y=config.get('column'),
            title=config.get('title', 'Box Plot')
        )
    else:
        return px.box(
            df,
            y=config.get('column'),
            title=config.get('title', 'Box Plot')
        )


def create_histogram(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    return px.histogram(
        df,
        x=config.get('column'),
        title=config.get('title', 'Histogram'),
        nbins=30
    )


def create_area_chart(config: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    return px.area(
        df,
        x=config.get('x'),
        y=config.get('y'),
        color=config.get('color'),
        title=config.get('title', 'Area Chart')
    )
