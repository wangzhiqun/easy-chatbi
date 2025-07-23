"""
Main Streamlit application for ChatBI platform.
Provides the web interface for natural language data analysis.
"""

import streamlit as st
import asyncio
from typing import Dict, Any
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import settings
from utils.logger import get_logger
from ui.utils.api_client import APIClient
from ui.utils.helpers import initialize_session_state, handle_authentication

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="ChatBI - Intelligent Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/chatbi/help',
        'Report a bug': 'https://github.com/chatbi/issues',
        'About': f"ChatBI v{settings.app_version} - Intelligent Data Analytics Platform"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .chat-container {
        background: #f9fafb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
    }

    .success-message {
        background: #dcfce7;
        border: 1px solid #16a34a;
        border-radius: 0.5rem;
        padding: 0.75rem;
        color: #15803d;
        margin: 1rem 0;
    }

    .error-message {
        background: #fef2f2;
        border: 1px solid #ef4444;
        border-radius: 0.5rem;
        padding: 0.75rem;
        color: #dc2626;
        margin: 1rem 0;
    }

    .info-message {
        background: #dbeafe;
        border: 1px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 0.75rem;
        color: #1d4ed8;
        margin: 1rem 0;
    }

    .sidebar .sidebar-content {
        background: #f8fafc;
    }

    .query-result {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }

    .sql-code {
        background: #1f2937;
        color: #f9fafb;
        border-radius: 0.375rem;
        padding: 1rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.875rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()

        # Handle authentication
        if not handle_authentication():
            return

        # Main application header
        st.markdown('<h1 class="main-header">📊 ChatBI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Intelligent Data Analytics through Natural Language</p>',
                    unsafe_allow_html=True)

        # Sidebar navigation
        with st.sidebar:
            st.title("🧭 Navigation")

            # Display current user
            if st.session_state.get('user'):
                st.success(f"👤 Welcome, {st.session_state.user.get('username', 'User')}")

            st.markdown("---")

            # Quick stats
            st.subheader("📈 Quick Stats")

            # Get stats from API
            api_client = APIClient()
            try:
                stats = asyncio.run(api_client.get_data_statistics())

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", stats.get('total_queries', 0))
                with col2:
                    st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")

                st.metric("Available Tables", stats.get('table_count', 0))

            except Exception as e:
                st.error(f"Failed to load stats: {e}")

            st.markdown("---")

            # Navigation links
            st.subheader("🔗 Quick Links")
            if st.button("📊 Overview Dashboard", use_container_width=True):
                st.switch_page("pages/1_📊_Overview.py")

            if st.button("💬 Chat Interface", use_container_width=True):
                st.switch_page("pages/2_💬_Chat.py")

            st.markdown("---")

            # System status
            st.subheader("🔋 System Status")
            try:
                health = asyncio.run(api_client.get_health_status())
                if health.get('status') == 'healthy':
                    st.success("✅ All systems operational")
                else:
                    st.warning("⚠️ Some services may be unavailable")
            except Exception:
                st.error("❌ Unable to check system status")

            # Settings and logout
            st.markdown("---")
            if st.button("⚙️ Settings", use_container_width=True):
                show_settings()

            if st.button("🚪 Logout", use_container_width=True):
                logout_user()

        # Main content area
        main_content()

        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
                ChatBI v{settings.app_version} | 
                <a href="https://github.com/chatbi" style="color: #3b82f6;">GitHub</a> | 
                <a href="mailto:support@chatbi.com" style="color: #3b82f6;">Support</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {e}")


def main_content():
    """Render main content area."""
    # Welcome message and quick start
    st.markdown("""
    <div class="info-message">
        <h3>🚀 Welcome to ChatBI!</h3>
        <p>Start by asking questions about your data in natural language. 
        ChatBI will automatically generate SQL queries and provide insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick start section
    st.subheader("🎯 Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💬 Ask a Question")
        st.markdown("""
        - "How many customers do we have?"
        - "What are the top 5 products by sales?"
        - "Show me revenue trends for the last 6 months"
        """)

        if st.button("Start Chatting →", use_container_width=True, type="primary"):
            st.switch_page("pages/2_💬_Chat.py")

    with col2:
        st.markdown("### 📊 Explore Data")
        st.markdown("""
        - View available tables and columns
        - Browse sample data
        - Generate automatic insights
        """)

        if st.button("View Dashboard →", use_container_width=True):
            st.switch_page("pages/1_📊_Overview.py")

    # Sample queries section
    st.subheader("💡 Example Questions")

    example_queries = [
        "Show me the total sales for each month this year",
        "Which customers have placed the most orders?",
        "What's the average order value by product category?",
        "Compare revenue between different regions",
        "Find customers who haven't ordered in the last 90 days"
    ]

    for i, query in enumerate(example_queries):
        if st.button(f"📝 {query}", key=f"example_{i}", use_container_width=True):
            # Store the query in session state and navigate to chat
            st.session_state.example_query = query
            st.switch_page("pages/2_💬_Chat.py")

    # Recent activity section
    st.subheader("📋 Recent Activity")

    try:
        api_client = APIClient()
        recent_queries = asyncio.run(api_client.get_recent_queries(limit=5))

        if recent_queries:
            for query in recent_queries:
                with st.expander(f"🔍 {query.get('question', 'Unknown query')[:60]}...", expanded=False):
                    st.write(f"**Status:** {query.get('execution_status', 'Unknown')}")
                    st.write(f"**Date:** {query.get('created_at', 'Unknown')}")
                    if query.get('result_rows'):
                        st.write(f"**Results:** {query.get('result_rows')} rows")
        else:
            st.info("No recent queries found. Start by asking a question!")

    except Exception as e:
        st.warning(f"Could not load recent activity: {e}")

    # Help section
    with st.expander("❓ Need Help?", expanded=False):
        st.markdown("""
        ### Getting Started
        1. **Ask Natural Questions**: Type your questions in plain English
        2. **Review Generated SQL**: See the SQL query that ChatBI created
        3. **Explore Results**: View data in tables and charts
        4. **Refine Your Questions**: Ask follow-up questions for deeper insights

        ### Tips for Better Results
        - Be specific about what data you want
        - Mention table names if you know them
        - Ask for specific time periods or filters
        - Use follow-up questions to drill down

        ### Example Patterns
        - **Counting**: "How many [items] do we have?"
        - **Trends**: "Show [metric] over time"
        - **Comparisons**: "Compare [A] vs [B]"
        - **Top/Bottom**: "What are the top 10 [items] by [metric]?"
        """)


def show_settings():
    """Show settings modal."""
    st.subheader("⚙️ Settings")

    # User preferences
    st.markdown("### 👤 User Preferences")

    # Theme selection
    theme = st.selectbox(
        "Theme",
        ["Auto", "Light", "Dark"],
        index=0,
        help="Choose your preferred color theme"
    )

    # Query settings
    st.markdown("### 🔍 Query Settings")

    auto_execute = st.checkbox(
        "Auto-execute queries",
        value=True,
        help="Automatically execute generated SQL queries"
    )

    max_results = st.slider(
        "Maximum results per query",
        min_value=10,
        max_value=1000,
        value=100,
        help="Limit the number of rows returned"
    )

    # Visualization settings
    st.markdown("### 📊 Visualization Settings")

    auto_charts = st.checkbox(
        "Auto-generate charts",
        value=True,
        help="Automatically suggest chart visualizations"
    )

    chart_theme = st.selectbox(
        "Chart theme",
        ["Streamlit", "Plotly", "Plotly Dark"],
        index=0
    )

    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        # In a real app, save these to user preferences
        st.success("Settings saved successfully!")


def logout_user():
    """Handle user logout."""
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.success("You have been logged out successfully!")
    st.rerun()


if __name__ == "__main__":
    main()