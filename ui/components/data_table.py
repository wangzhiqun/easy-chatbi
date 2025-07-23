"""
Data table components for ChatBI platform.
Enhanced data display with filtering, sorting, and export capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import io
from datetime import datetime


def render_data_table(
        df: pd.DataFrame,
        title: Optional[str] = None,
        max_rows: int = 100,
        show_filters: bool = True,
        show_export: bool = True,
        show_stats: bool = True,
        key_suffix: str = ""
):
    """
    Render an enhanced data table with filtering and export options.

    Args:
        df: DataFrame to display
        title: Optional title for the table
        max_rows: Maximum number of rows to display
        show_filters: Whether to show column filters
        show_export: Whether to show export buttons
        show_stats: Whether to show data statistics
        key_suffix: Suffix for unique widget keys
    """

    if df.empty:
        st.info("No data to display")
        return

    # Title
    if title:
        st.subheader(title)

    # Data statistics
    if show_stats:
        render_table_stats(df, key_suffix)

    # Filters
    filtered_df = df.copy()
    if show_filters and len(df) > 10:
        filtered_df = render_table_filters(df, key_suffix)

    # Pagination
    if len(filtered_df) > max_rows:
        filtered_df = render_pagination(filtered_df, max_rows, key_suffix)

    # Display table
    display_enhanced_table(filtered_df, key_suffix)

    # Export options
    if show_export:
        render_export_options(filtered_df, title or "data", key_suffix)


def render_table_stats(df: pd.DataFrame, key_suffix: str = ""):
    """Render table statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Rows", f"{len(df):,}")

    with col2:
        st.metric("📋 Columns", len(df.columns))

    with col3:
        null_count = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        completeness = 100 - (null_count / total_cells * 100) if total_cells > 0 else 100
        st.metric("✅ Completeness", f"{completeness:.1f}%")

    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        st.metric("💾 Memory", f"{memory_usage:.1f} MB")


def render_table_filters(df: pd.DataFrame, key_suffix: str = "") -> pd.DataFrame:
    """Render column filters and return filtered DataFrame."""

    with st.expander("🔍 Filters", expanded=False):
        filtered_df = df.copy()

        # Create columns for filters
        filter_cols = st.columns(min(len(df.columns), 4))

        for i, column in enumerate(df.columns):
            with filter_cols[i % len(filter_cols)]:
                column_type = df[column].dtype

                if column_type in ['object', 'category']:
                    # Categorical filter
                    unique_values = df[column].dropna().unique()
                    if len(unique_values) <= 50:  # Only show filter for reasonable number of values
                        selected_values = st.multiselect(
                            f"Filter {column}",
                            options=sorted(unique_values.astype(str)),
                            default=None,
                            key=f"filter_{column}_{key_suffix}"
                        )

                        if selected_values:
                            filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected_values)]

                elif column_type in ['int64', 'float64']:
                    # Numeric range filter
                    min_val = float(df[column].min())
                    max_val = float(df[column].max())

                    if min_val != max_val:
                        range_values = st.slider(
                            f"Range {column}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"range_{column}_{key_suffix}"
                        )

                        filtered_df = filtered_df[
                            (filtered_df[column] >= range_values[0]) &
                            (filtered_df[column] <= range_values[1])
                            ]

                elif 'datetime' in str(column_type):
                    # Date range filter
                    min_date = df[column].min()
                    max_date = df[column].max()

                    if pd.notna(min_date) and pd.notna(max_date) and min_date != max_date:
                        date_range = st.date_input(
                            f"Date range {column}",
                            value=(min_date.date(), max_date.date()),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"date_{column}_{key_suffix}"
                        )

                        if len(date_range) == 2:
                            start_date, end_date = date_range
                            filtered_df = filtered_df[
                                (filtered_df[column].dt.date >= start_date) &
                                (filtered_df[column].dt.date <= end_date)
                                ]

        # Text search across all string columns
        search_term = st.text_input(
            "🔍 Search across all text columns",
            placeholder="Enter search term...",
            key=f"search_{key_suffix}"
        )

        if search_term:
            text_columns = df.select_dtypes(include=['object']).columns
            mask = pd.Series([False] * len(filtered_df))

            for col in text_columns:
                mask |= filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)

            filtered_df = filtered_df[mask]

        # Show filter summary
        if len(filtered_df) != len(df):
            st.info(f"Showing {len(filtered_df):,} of {len(df):,} rows")

    return filtered_df


def render_pagination(df: pd.DataFrame, max_rows: int, key_suffix: str = "") -> pd.DataFrame:
    """Render pagination controls and return paginated DataFrame."""

    total_rows = len(df)
    total_pages = (total_rows + max_rows - 1) // max_rows

    if total_pages <= 1:
        return df

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        page = st.selectbox(
            f"Page (showing {max_rows} rows per page)",
            options=list(range(1, total_pages + 1)),
            index=0,
            key=f"page_{key_suffix}"
        )

    start_idx = (page - 1) * max_rows
    end_idx = min(start_idx + max_rows, total_rows)

    st.info(f"Showing rows {start_idx + 1:,} to {end_idx:,} of {total_rows:,}")

    return df.iloc[start_idx:end_idx]


def display_enhanced_table(df: pd.DataFrame, key_suffix: str = ""):
    """Display table with enhanced formatting."""

    if df.empty:
        st.info("No data to display after filtering")
        return

    # Format the DataFrame for better display
    display_df = format_dataframe_for_display(df)

    # Display with custom styling
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(400, (len(display_df) + 1) * 35),  # Dynamic height
        column_config=get_column_config(df)
    )

    # Column information
    with st.expander("📋 Column Information", expanded=False):
        render_column_info(df)


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame for better display in Streamlit."""
    display_df = df.copy()

    for column in display_df.columns:
        column_type = display_df[column].dtype

        # Format numeric columns
        if column_type in ['float64']:
            # Round floats to 2 decimal places
            display_df[column] = display_df[column].round(2)

        elif column_type in ['int64']:
            # Format large integers with commas
            display_df[column] = display_df[column].apply(
                lambda x: f"{x:,}" if pd.notna(x) and abs(x) >= 1000 else x
            )

        # Format datetime columns
        elif 'datetime' in str(column_type):
            display_df[column] = display_df[column].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Truncate long text
        elif column_type == 'object':
            display_df[column] = display_df[column].astype(str).apply(
                lambda x: x[:100] + "..." if len(str(x)) > 100 else x
            )

    return display_df


def get_column_config(df: pd.DataFrame) -> Dict[str, Any]:
    """Get column configuration for Streamlit dataframe display."""
    config = {}

    for column in df.columns:
        column_type = df[column].dtype

        if column_type in ['int64', 'float64']:
            config[column] = st.column_config.NumberColumn(
                column,
                help=f"Numeric column: {column}",
                format="%.2f" if column_type == 'float64' else "%d"
            )

        elif 'datetime' in str(column_type):
            config[column] = st.column_config.DatetimeColumn(
                column,
                help=f"Date/time column: {column}",
                format="YYYY-MM-DD HH:mm:ss"
            )

        elif column_type == 'object':
            # Check if it might be a URL
            sample_values = df[column].dropna().head(5).astype(str)
            if any(val.startswith(('http://', 'https://')) for val in sample_values):
                config[column] = st.column_config.LinkColumn(
                    column,
                    help=f"Link column: {column}"
                )
            else:
                config[column] = st.column_config.TextColumn(
                    column,
                    help=f"Text column: {column}",
                    max_chars=100
                )

    return config


def render_column_info(df: pd.DataFrame):
    """Render detailed column information."""

    column_info = []

    for column in df.columns:
        col_data = df[column]

        info = {
            'Column': column,
            'Type': str(col_data.dtype),
            'Non-Null Count': col_data.count(),
            'Null Count': col_data.isnull().sum(),
            'Unique Values': col_data.nunique()
        }

        # Add type-specific info
        if col_data.dtype in ['int64', 'float64']:
            info.update({
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Mean': col_data.mean(),
                'Std Dev': col_data.std()
            })

        elif col_data.dtype == 'object':
            # Most common values
            top_values = col_data.value_counts().head(3)
            info['Top Values'] = ', '.join([f"{val} ({count})" for val, count in top_values.items()])

        column_info.append(info)

    info_df = pd.DataFrame(column_info)
    st.dataframe(info_df, use_container_width=True)


def render_export_options(df: pd.DataFrame, filename_base: str = "data", key_suffix: str = ""):
    """Render export options for the data."""

    st.markdown("### 📥 Export Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📄 CSV", key=f"csv_{key_suffix}", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📊 Excel", key=f"excel_{key_suffix}", use_container_width=True):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)

            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col3:
        if st.button("📋 JSON", key=f"json_{key_suffix}", use_container_width=True):
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col4:
        if st.button("🖨️ Print View", key=f"print_{key_suffix}", use_container_width=True):
            render_print_view(df, filename_base)


def render_print_view(df: pd.DataFrame, title: str):
    """Render a print-friendly view of the data."""

    st.markdown("### 🖨️ Print Preview")

    # Create a clean, print-friendly format
    st.markdown(f"**{title}**")
    st.markdown(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    st.markdown(f"*Total rows: {len(df):,}*")

    # Display table without interactive features
    st.table(df.head(50))  # Limit to 50 rows for print view

    if len(df) > 50:
        st.info(f"Showing first 50 of {len(df):,} rows")


def render_data_summary(df: pd.DataFrame):
    """Render comprehensive data summary."""

    st.subheader("📊 Data Summary")

    # Basic info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Basic Information")
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.write(f"**Memory usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # Data types
        dtype_counts = df.dtypes.value_counts()
        st.write("**Data types:**")
        for dtype, count in dtype_counts.items():
            st.write(f"  • {dtype}: {count} columns")

    with col2:
        st.markdown("#### 🔍 Data Quality")

        # Missing data
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df) * 100).round(1)

        if missing_data.sum() > 0:
            st.write("**Columns with missing data:**")
            for col, missing_count in missing_data[missing_data > 0].items():
                st.write(f"  • {col}: {missing_count} ({missing_percentage[col]}%)")
        else:
            st.write("✅ No missing data found")

        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.write(f"⚠️ **Duplicate rows:** {duplicate_count}")
        else:
            st.write("✅ No duplicate rows found")

    # Numeric summary
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.markdown("#### 📈 Numeric Columns Summary")
        st.dataframe(numeric_df.describe(), use_container_width=True)

    # Categorical summary
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        st.markdown("#### 📝 Categorical Columns Summary")

        cat_summary = []
        for col in categorical_df.columns:
            cat_summary.append({
                'Column': col,
                'Unique Values': df[col].nunique(),
                'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'Frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0
            })

        cat_df = pd.DataFrame(cat_summary)
        st.dataframe(cat_df, use_container_width=True)


def render_interactive_table(
        df: pd.DataFrame,
        title: Optional[str] = None,
        searchable_columns: Optional[List[str]] = None,
        sortable: bool = True,
        key_suffix: str = ""
):
    """
    Render an interactive table with advanced features.

    Args:
        df: DataFrame to display
        title: Optional title
        searchable_columns: Columns to include in search
        sortable: Whether columns are sortable
        key_suffix: Unique key suffix
    """

    if title:
        st.subheader(title)

    # Search functionality
    if searchable_columns:
        search_term = st.text_input(
            "🔍 Search",
            placeholder=f"Search in: {', '.join(searchable_columns)}",
            key=f"search_interactive_{key_suffix}"
        )

        if search_term:
            mask = pd.Series([False] * len(df))
            for col in searchable_columns:
                if col in df.columns:
                    mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
            df = df[mask]

    # Sorting
    if sortable and not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            sort_column = st.selectbox(
                "Sort by",
                options=[''] + list(df.columns),
                key=f"sort_col_{key_suffix}"
            )

        with col2:
            sort_order = st.selectbox(
                "Order",
                options=['Ascending', 'Descending'],
                key=f"sort_order_{key_suffix}"
            )

        if sort_column:
            ascending = sort_order == 'Ascending'
            df = df.sort_values(by=sort_column, ascending=ascending)

    # Display table
    st.dataframe(df, use_container_width=True)


def create_comparison_table(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        title1: str = "Dataset 1",
        title2: str = "Dataset 2"
):
    """Create a side-by-side comparison of two datasets."""

    st.subheader("🔄 Dataset Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {title1}")
        st.dataframe(df1.head(10), use_container_width=True)
        st.write(f"Shape: {df1.shape}")

    with col2:
        st.markdown(f"#### {title2}")
        st.dataframe(df2.head(10), use_container_width=True)
        st.write(f"Shape: {df2.shape}")

    # Comparison metrics
    st.markdown("#### 📊 Comparison Metrics")

    comparison_data = {
        'Metric': ['Rows', 'Columns', 'Memory (MB)', 'Numeric Columns', 'Text Columns'],
        title1: [
            f"{df1.shape[0]:,}",
            df1.shape[1],
            f"{df1.memory_usage(deep=True).sum() / 1024 / 1024:.1f}",
            len(df1.select_dtypes(include=[np.number]).columns),
            len(df1.select_dtypes(include=['object']).columns)
        ],
        title2: [
            f"{df2.shape[0]:,}",
            df2.shape[1],
            f"{df2.memory_usage(deep=True).sum() / 1024 / 1024:.1f}",
            len(df2.select_dtypes(include=[np.number]).columns),
            len(df2.select_dtypes(include=['object']).columns)
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)