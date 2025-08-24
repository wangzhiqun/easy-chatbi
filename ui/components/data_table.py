from typing import Optional

import pandas as pd
import streamlit as st


def render_data_table(
        df: pd.DataFrame,
        title: Optional[str] = None,
        show_stats: bool = True,
        show_export: bool = True,
        editable: bool = False
):
    if title:
        st.subheader(title)

    if show_stats:
        render_table_stats(df)

    filtered_df = apply_filters(df)

    if editable:
        edited_df = st.data_editor(
            filtered_df,
            use_container_width=True,
            num_rows="dynamic"
        )

        if st.button("💾 Save Changes"):
            st.success("Changes saved!")
            return edited_df
    else:
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

    if show_export:
        render_export_options(filtered_df)

    return filtered_df


def render_table_stats(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_usage:.2f} MB")

    with col4:
        null_count = df.isnull().sum().sum()
        st.metric("Missing Values", f"{null_count:,}")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔍 Filters"):
        filtered_df = df.copy()

        selected_columns = st.multiselect(
            "Select Columns",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

        if selected_columns:
            filtered_df = filtered_df[selected_columns]

        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Filters")

                for col in numeric_cols[:3]:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())

                    range_vals = st.slider(
                        f"{col}",
                        min_val,
                        max_val,
                        (min_val, max_val)
                    )

                    filtered_df = filtered_df[
                        (filtered_df[col] >= range_vals[0]) &
                        (filtered_df[col] <= range_vals[1])
                        ]

        with col2:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                st.subheader("Categorical Filters")

                for col in categorical_cols[:3]:
                    unique_vals = df[col].unique()

                    selected_vals = st.multiselect(
                        f"{col}",
                        options=unique_vals,
                        default=unique_vals
                    )

                    if selected_vals:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

        search_term = st.text_input("🔍 Search in all columns")
        if search_term:
            mask = filtered_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

        if len(filtered_df) < len(df):
            st.info(f"Showing {len(filtered_df)} of {len(df)} rows")

    return filtered_df


def render_export_options(df: pd.DataFrame):
    with st.expander("📥 Export Options"):
        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name="data_export.csv",
                mime="text/csv"
            )

        with col2:
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_str,
                file_name="data_export.json",
                mime="application/json"
            )

        with col3:
            if st.button("📊 Download Excel"):
                import io
                buffer = io.BytesIO()

                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)

                st.download_button(
                    label="📊 Download",
                    data=buffer.getvalue(),
                    file_name="data_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


def render_data_summary(df: pd.DataFrame):
    st.subheader("📊 Data Summary")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistics", "Missing Data", "Correlations"])

    with tab1:
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", ", ".join(df.columns))

        st.write("**Data Types:**")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Type'])
        st.dataframe(dtype_df)

    with tab2:
        st.write("**Descriptive Statistics:**")
        st.dataframe(df.describe())

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            st.write("**Categorical Statistics:**")
            for col in cat_cols:
                st.write(f"- {col}: {df[col].nunique()} unique values")

    with tab3:
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })

        missing_data = missing_data[missing_data['Missing Count'] > 0]

        if len(missing_data) > 0:
            st.dataframe(missing_data)
        else:
            st.success("No missing data!")

    with tab4:
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            import plotly.express as px
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu'
            )

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis")
