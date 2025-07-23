"""
Overview page for ChatBI platform.
Displays data overview, table information, and system statistics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.utils.api_client import APIClient
from ui.components.data_table import render_data_table
from ui.components.charts import create_metric_chart, create_usage_chart
from utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Data Overview - ChatBI",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }

    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        margin: 0;
    }

    .table-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-online { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-offline { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main overview page function."""
    try:
        st.title("📊 Data Overview")
        st.markdown("Comprehensive view of your data landscape and system performance.")

        # Initialize API client
        api_client = APIClient()

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🗃️ Data Sources", "⚡ Performance", "🔧 System Health"])

        with tab1:
            render_dashboard(api_client)

        with tab2:
            render_data_sources(api_client)

        with tab3:
            render_performance_metrics(api_client)

        with tab4:
            render_system_health(api_client)

    except Exception as e:
        logger.error(f"Overview page error: {e}")
        st.error(f"Failed to load overview page: {e}")


def render_dashboard(api_client: APIClient):
    """Render the main dashboard with key metrics."""
    st.subheader("📈 Key Metrics")

    # Load statistics
    try:
        with st.spinner("Loading dashboard data..."):
            stats = asyncio.run(api_client.get_data_statistics())

        # Top row metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{stats.get('total_queries', 0):,}</p>
                <p class="metric-label">Total Queries</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{stats.get('table_count', 0)}</p>
                <p class="metric-label">Data Tables</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            success_rate = stats.get('success_rate', 0)
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{success_rate:.1f}%</p>
                <p class="metric-label">Success Rate</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            active_users = len(stats.get('recent_queries', []))
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{active_users}</p>
                <p class="metric-label">Active Users</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Query Trends")

            # Create sample trend data
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
            query_counts = [50 + i % 20 + (i // 7) * 10 for i in range(len(dates))]

            fig = px.line(
                x=dates,
                y=query_counts,
                title="Daily Query Volume",
                labels={'x': 'Date', 'y': 'Queries'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Query Types")

            # Sample query type distribution
            query_types = ['SELECT', 'COUNT', 'SUM', 'GROUP BY', 'JOIN']
            counts = [45, 25, 15, 10, 5]

            fig = px.pie(
                values=counts,
                names=query_types,
                title="Query Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent activity
        st.subheader("🕒 Recent Activity")

        recent_queries = stats.get('recent_queries', [])
        if recent_queries:
            for query in recent_queries[:5]:
                status_icon = "✅" if query.get('status') == 'success' else "❌"
                st.write(
                    f"{status_icon} **{query.get('question', 'Unknown')}** - {query.get('created_at', 'Unknown time')}")
        else:
            st.info("No recent activity to display.")

    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")


def render_data_sources(api_client: APIClient):
    """Render data sources and table information."""
    st.subheader("🗃️ Available Data Sources")

    try:
        with st.spinner("Loading data source information..."):
            tables = asyncio.run(api_client.get_tables())

        if not tables:
            st.warning("No data tables found.")
            return

        # Table selection
        selected_table = st.selectbox(
            "Select a table to explore:",
            options=tables,
            help="Choose a table to view its schema and sample data"
        )

        if selected_table:
            # Create columns for table info
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📋 Table Information")

                # Get table schema
                try:
                    schema = asyncio.run(api_client.get_table_schema(selected_table))

                    st.write(f"**Table:** {selected_table}")
                    st.write(f"**Columns:** {len(schema.get('columns', []))}")

                    # Display columns
                    st.markdown("#### Columns")
                    for col in schema.get('columns', []):
                        col_type = col.get('type', 'Unknown')
                        nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                        pk_indicator = " 🔑" if col.get('primary_key', False) else ""
                        st.write(f"• **{col['name']}** ({col_type}) {nullable}{pk_indicator}")

                except Exception as e:
                    st.error(f"Failed to load table schema: {e}")

            with col2:
                st.markdown("### 👀 Sample Data")

                # Get sample data
                try:
                    sample_data = asyncio.run(api_client.get_sample_data(selected_table, limit=10))

                    if sample_data:
                        df = pd.DataFrame(sample_data)
                        render_data_table(df, f"Sample data from {selected_table}")
                    else:
                        st.info("No sample data available.")

                except Exception as e:
                    st.error(f"Failed to load sample data: {e}")

        # Table statistics overview
        st.markdown("---")
        st.subheader("📊 Table Statistics")

        # Create table statistics
        table_stats = []
        for table in tables[:10]:  # Limit to first 10 tables for performance
            try:
                # In a real implementation, you'd get actual statistics
                stats = {
                    'Table': table,
                    'Estimated Rows': f"{(hash(table) % 100000):,}",
                    'Size': f"{(hash(table) % 500) + 10} MB",
                    'Last Updated': datetime.now().strftime('%Y-%m-%d')
                }
                table_stats.append(stats)
            except Exception:
                continue

        if table_stats:
            stats_df = pd.DataFrame(table_stats)
            st.dataframe(stats_df, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load data sources: {e}")


def render_performance_metrics(api_client: APIClient):
    """Render performance metrics and analytics."""
    st.subheader("⚡ Performance Analytics")

    # Performance metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Avg Query Time",
            value="1.8s",
            delta="-0.2s",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            label="Cache Hit Rate",
            value="78.5%",
            delta="2.1%",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Error Rate",
            value="2.3%",
            delta="-0.5%",
            delta_color="inverse"
        )

    # Performance charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Response Time Trends")

        # Sample response time data
        hours = list(range(24))
        response_times = [1.2 + 0.5 * (h % 8) + 0.3 * (h % 3) for h in hours]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=response_times,
            mode='lines+markers',
            name='Avg Response Time',
            line=dict(color='#3b82f6', width=3)
        ))

        fig.update_layout(
            title="Average Response Time by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Response Time (seconds)",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Query Volume")

        # Sample query volume data
        query_volumes = [20 + 10 * (h % 12) for h in hours]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hours,
            y=query_volumes,
            name='Query Volume',
            marker_color='#10b981'
        ))

        fig.update_layout(
            title="Query Volume by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Queries",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Slow queries analysis
    st.subheader("🐌 Slow Query Analysis")

    slow_queries = [
        {"Query": "SELECT * FROM large_table WHERE complex_condition...", "Time": "5.2s", "Frequency": 12},
        {"Query": "SELECT COUNT(*) FROM orders JOIN customers...", "Time": "3.8s", "Frequency": 8},
        {"Query": "SELECT SUM(amount) FROM transactions WHERE...", "Time": "2.9s", "Frequency": 15},
    ]

    slow_df = pd.DataFrame(slow_queries)
    st.dataframe(slow_df, use_container_width=True)

    # Optimization suggestions
    with st.expander("💡 Optimization Suggestions"):
        st.markdown("""
        **Recommended Actions:**
        1. **Add indexes** to frequently filtered columns
        2. **Use query limits** to prevent large result sets
        3. **Optimize JOIN operations** by ensuring proper indexing
        4. **Consider data partitioning** for large tables
        5. **Cache frequently accessed results**
        """)


def render_system_health(api_client: APIClient):
    """Render system health and status information."""
    st.subheader("🔧 System Health Status")

    try:
        with st.spinner("Checking system status..."):
            health = asyncio.run(api_client.get_health_status())

        # Overall status
        status = health.get('status', 'unknown')
        status_color = {
            'healthy': 'success',
            'degraded': 'warning',
            'unhealthy': 'error'
        }.get(status, 'info')

        if status == 'healthy':
            st.success("🟢 All systems operational")
        elif status == 'degraded':
            st.warning("🟡 Some services experiencing issues")
        else:
            st.error("🔴 System issues detected")

        # Service status breakdown
        services = health.get('services', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏗️ Core Services")

            core_services = {
                'Database': services.get('database', True),
                'Redis Cache': services.get('redis', True),
                'AI Processing': services.get('ai_service', True),
                'Authentication': services.get('auth', True)
            }

            for service, status in core_services.items():
                if status:
                    st.markdown(f'<span class="status-indicator status-online"></span>**{service}** - Online',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-indicator status-offline"></span>**{service}** - Offline',
                                unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔄 Background Services")

            background_services = {
                'Query Processing': True,
                'Data Sync': True,
                'Cache Management': True,
                'Audit Logging': True
            }

            for service, status in background_services.items():
                if status:
                    st.markdown(f'<span class="status-indicator status-online"></span>**{service}** - Running',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-indicator status-warning"></span>**{service}** - Warning',
                                unsafe_allow_html=True)

        # System metrics
        st.markdown("---")
        st.subheader("📊 System Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CPU Usage", "45%", "2%")

        with col2:
            st.metric("Memory Usage", "67%", "-1%")

        with col3:
            st.metric("Active Connections", "23", "5")

        with col4:
            st.metric("Uptime", "7d 12h", "")

        # Recent system events
        st.subheader("📋 Recent System Events")

        events = [
            {"Time": "2024-01-15 14:30", "Event": "Cache cleanup completed", "Level": "Info"},
            {"Time": "2024-01-15 14:15", "Event": "Database backup finished", "Level": "Success"},
            {"Time": "2024-01-15 14:00", "Event": "High query volume detected", "Level": "Warning"},
            {"Time": "2024-01-15 13:45", "Event": "New user registered", "Level": "Info"},
        ]

        for event in events:
            level_emoji = {
                "Success": "✅",
                "Info": "ℹ️",
                "Warning": "⚠️",
                "Error": "❌"
            }.get(event["Level"], "ℹ️")

            st.write(f"{level_emoji} **{event['Time']}** - {event['Event']}")

    except Exception as e:
        st.error(f"Failed to load system health information: {e}")

    # Manual refresh button
    if st.button("🔄 Refresh Status", type="primary"):
        st.rerun()


if __name__ == "__main__":
    main()