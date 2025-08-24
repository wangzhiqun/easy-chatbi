import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_sample_data() -> pd.DataFrame:
    try:
        df = pd.read_csv('data/sample.csv')
    except:

        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

        data = {
            'date': dates,
            'product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor'], 30),
            'category': np.random.choice(['Electronics', 'Accessories'], 30),
            'quantity': np.random.randint(1, 50, 30),
            'revenue': np.random.uniform(100, 5000, 30).round(2),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 30)
        }

        df = pd.DataFrame(data)

    return df


def save_conversation(conversation_id: str, messages: List[Dict[str, Any]]) -> bool:
    import streamlit as st
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{conversation_id}_{timestamp}.json"

        conversation_data = {
            "conversation_id": conversation_id,
            "messages": messages,
            "timestamp": timestamp
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        st.error(f"Failed to save conversation: {str(e)}")
        return False


def format_sql_query(query: str) -> str:
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'INNER JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT',
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP'
    ]

    formatted = query
    for keyword in keywords:
        formatted = formatted.replace(keyword, f"\n{keyword}")
        formatted = formatted.replace(keyword.lower(), f"\n{keyword}")

    lines = [line.strip() for line in formatted.split('\n') if line.strip()]
    return '\n'.join(lines)


def render_chart(config: Dict[str, Any], df: pd.DataFrame):
    chart_type = config.get('type', 'bar')
    import streamlit as st

    try:
        if chart_type == 'line':
            fig = px.line(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', 'Line Chart')
            )

        elif chart_type == 'bar':
            fig = px.bar(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', 'Bar Chart')
            )

        elif chart_type == 'scatter':
            fig = px.scatter(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                size=config.get('size'),
                title=config.get('title', 'Scatter Plot')
            )

        elif chart_type == 'pie':
            fig = px.pie(
                df,
                values=config.get('values'),
                names=config.get('names'),
                title=config.get('title', 'Pie Chart')
            )

        elif chart_type == 'heatmap':
            z_data = config.get('z')
            if isinstance(z_data, str) and z_data in df.columns:
                pivot_df = df.pivot_table(
                    index=config.get('y'),
                    columns=config.get('x'),
                    values=z_data,
                    aggfunc='mean'
                )
                z_data = pivot_df.values

            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=config.get('x'),
                y=config.get('y'),
                colorscale='Viridis'
            ))

            fig.update_layout(title=config.get('title', 'Heatmap'))

        elif chart_type == 'histogram':
            fig = px.histogram(
                df,
                x=config.get('x') or config.get('column'),
                title=config.get('title', 'Histogram'),
                nbins=config.get('nbins', 20)
            )

        elif chart_type == 'box':
            fig = px.box(
                df,
                y=config.get('y') or config.get('column'),
                x=config.get('x'),
                title=config.get('title', 'Box Plot')
            )

        elif chart_type == 'area':
            fig = px.area(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', 'Area Chart')
            )

        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return

        fig.update_layout(
            height=config.get('height', 400),
            showlegend=config.get('show_legend', True),
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")


def format_number(value: float, decimals: int = 2) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def get_color_scale(n: int) -> List[str]:
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    if n <= len(colors):
        return colors[:n]
    else:
        import colorsys
        additional = []
        for i in range(n - len(colors)):
            hue = i / (n - len(colors))
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = '#' + ''.join(f'{int(c * 255):02x}' for c in rgb)
            additional.append(hex_color)
        return colors + additional


def export_dataframe(df: pd.DataFrame, format: str = 'csv') -> bytes:
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8')
    elif format == 'excel':
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        return buffer.getvalue()
    else:
        raise ValueError(f"Unsupported format: {format}")


def parse_uploaded_file(file) -> Optional[pd.DataFrame]:
    import streamlit as st
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            return pd.read_excel(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file)
        else:
            st.error(f"Unsupported file type: {file.name}")
            return None
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    issues = {
        'warnings': [],
        'errors': [],
        'info': []
    }

    if df.empty:
        issues['errors'].append("DataFrame is empty")
        return issues

    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues['warnings'].append(f"Missing values in columns: {', '.join(missing_cols)}")

    if df.duplicated().any():
        issues['warnings'].append(f"Found {df.duplicated().sum()} duplicate rows")

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col])
                issues['info'].append(f"Column '{col}' could be converted to numeric")
            except:
                pass

    if len(df) > 10000:
        issues['info'].append(f"Large dataset with {len(df)} rows - consider sampling for visualization")

    return issues


def render_chart_from_metadata(metadata: Dict[str, Any], index: int):
    import streamlit as st

    try:
        chart_type = metadata.get('chart_type', 'table')
        chart_data = metadata.get('chart_data', {})
        chart_config = metadata.get('chart_config', {})

        st.markdown("---")
        st.subheader("📊 数据可视化")

        if chart_type == 'bar':
            render_bar_chart(chart_data, chart_config, f"bar_{index}")
        elif chart_type == 'line':
            render_line_chart(chart_data, chart_config, f"line_{index}")
        elif chart_type == 'pie':
            render_pie_chart(chart_data, chart_config, f"pie_{index}")
        elif chart_type == 'scatter':
            render_scatter_chart(chart_data, chart_config, f"scatter_{index}")
        else:
            render_table_chart(chart_data, chart_config, f"table_{index}")

        add_chart_download_options(chart_data, chart_config, index)

    except Exception as e:
        st.error(f"渲染图表时出错：{str(e)}")


def render_bar_chart(chart_data: Dict, chart_config: Dict, key: str):
    import streamlit as st

    try:
        labels = chart_data.get('labels', [])
        datasets = chart_data.get('datasets', [])

        if datasets and len(datasets) > 0:
            dataset = datasets[0]
            data_values = dataset.get('data', [])

            df = pd.DataFrame({
                'x': labels,
                'y': data_values
            })

            fig = px.bar(
                df,
                x='x',
                y='y',
                title=chart_config.get('title', '柱状图'),
                labels={
                    'x': chart_config.get('x_title', 'X轴'),
                    'y': chart_config.get('y_title', 'Y轴')
                }
            )

            fig.update_layout(
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True, key=key)
        else:
            st.warning("没有可用的图表数据")

    except Exception as e:
        st.error(f"创建柱状图时出错：{str(e)}")


def render_line_chart(chart_data: Dict, chart_config: Dict, key: str):
    import streamlit as st

    try:
        labels = chart_data.get('labels', [])
        datasets = chart_data.get('datasets', [])

        if datasets and len(datasets) > 0:
            dataset = datasets[0]
            data_values = dataset.get('data', [])

            df = pd.DataFrame({
                'x': labels,
                'y': data_values
            })

            fig = px.line(
                df,
                x='x',
                y='y',
                title=chart_config.get('title', '折线图'),
                labels={
                    'x': chart_config.get('x_title', 'X轴'),
                    'y': chart_config.get('y_title', 'Y轴')
                }
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=key)
        else:
            st.warning("没有可用的图表数据")

    except Exception as e:
        st.error(f"创建折线图时出错：{str(e)}")


def render_pie_chart(chart_data: Dict, chart_config: Dict, key: str):
    import streamlit as st

    try:
        labels = chart_data.get('labels', [])
        datasets = chart_data.get('datasets', [])

        if datasets and len(datasets) > 0:
            dataset = datasets[0]
            values = dataset.get('data', [])

            df = pd.DataFrame({
                'labels': labels,
                'values': values
            })

            fig = px.pie(
                df,
                values='values',
                names='labels',
                title=chart_config.get('title', '饼图')
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=key)
        else:
            st.warning("没有可用的图表数据")

    except Exception as e:
        st.error(f"创建饼图时出错：{str(e)}")


def render_scatter_chart(chart_data: Dict, chart_config: Dict, key: str):
    import streamlit as st

    try:
        labels = chart_data.get('labels', [])
        datasets = chart_data.get('datasets', [])

        if datasets and len(datasets) > 0:
            dataset = datasets[0]
            data_values = dataset.get('data', [])

            df = pd.DataFrame({
                'x': labels,
                'y': data_values
            })

            fig = px.scatter(
                df,
                x='x',
                y='y',
                title=chart_config.get('title', '散点图'),
                labels={
                    'x': chart_config.get('x_title', 'X轴'),
                    'y': chart_config.get('y_title', 'Y轴')
                }
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=key)
        else:
            st.warning("没有可用的图表数据")

    except Exception as e:
        st.error(f"创建散点图时出错：{str(e)}")


def render_table_chart(chart_data: Dict, chart_config: Dict, key: str):
    import streamlit as st

    try:
        if 'rows' in chart_data:
            df = pd.DataFrame(chart_data['rows'])
            st.subheader(chart_config.get('title', '数据表格'))
            st.dataframe(df, use_container_width=True, key=key)
        elif 'labels' in chart_data and 'datasets' in chart_data:
            labels = chart_data.get('labels', [])
            datasets = chart_data.get('datasets', [])

            if datasets:
                dataset = datasets[0]
                values = dataset.get('data', [])

                df = pd.DataFrame({
                    '标签': labels,
                    '数值': values
                })

                st.subheader(chart_config.get('title', '数据表格'))
                st.dataframe(df, use_container_width=True, key=key)
        else:
            st.warning("没有可用的表格数据")

    except Exception as e:
        st.error(f"创建表格时出错：{str(e)}")


def render_query_results(result: Dict[str, Any], index: int):
    import streamlit as st

    data = result.get('data', [])
    if data:
        st.markdown("---")
        st.subheader("📋 查询结果")
        df = pd.DataFrame(data)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("行数", len(df))
        with col2:
            st.metric("列数", len(df.columns))

        st.dataframe(df, use_container_width=True, key=f"query_result_{index}")

        with st.expander("快速可视化"):
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("生成柱状图", key=f"bar_quick_{index}"):
                    quick_visualization(df, 'bar', index)

            with col2:
                if st.button("生成折线图", key=f"line_quick_{index}"):
                    quick_visualization(df, 'line', index)

            with col3:
                if st.button("生成饼图", key=f"pie_quick_{index}"):
                    quick_visualization(df, 'pie', index)


def quick_visualization(df: pd.DataFrame, chart_type: str, index: int):
    import streamlit as st

    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

        if chart_type == 'bar' and numeric_cols and categorical_cols:
            fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            st.plotly_chart(fig, use_container_width=True, key=f"quick_bar_{index}")

        elif chart_type == 'line' and len(numeric_cols) >= 2:
            fig = px.line(df, x=numeric_cols[0], y=numeric_cols[1])
            st.plotly_chart(fig, use_container_width=True, key=f"quick_line_{index}")

        elif chart_type == 'pie' and numeric_cols and categorical_cols:
            grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().reset_index()
            fig = px.pie(grouped, values=numeric_cols[0], names=categorical_cols[0])
            st.plotly_chart(fig, use_container_width=True, key=f"quick_pie_{index}")

    except Exception as e:
        st.error(f"生成快速图表时出错：{str(e)}")


def add_chart_download_options(chart_data: Dict, chart_config: Dict, index: int):
    import streamlit as st

    with st.expander("下载选项"):
        col1, col2 = st.columns(2)

        with col1:
            if 'rows' in chart_data:
                df = pd.DataFrame(chart_data['rows'])
            else:
                labels = chart_data.get('labels', [])
                datasets = chart_data.get('datasets', [])
                if datasets:
                    values = datasets[0].get('data', [])
                    df = pd.DataFrame({'标签': labels, '数值': values})
                else:
                    df = pd.DataFrame()

            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "下载 CSV",
                    csv,
                    f"chart_data_{index}.csv",
                    "text/csv",
                    key=f"download_csv_{index}"
                )

        with col2:
            if not df.empty:
                json_str = df.to_json(orient='records', force_ascii=False)
                st.download_button(
                    "下载 JSON",
                    json_str,
                    f"chart_data_{index}.json",
                    "application/json",
                    key=f"download_json_{index}"
                )
