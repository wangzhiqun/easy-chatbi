"""
Helper utilities for ChatBI Streamlit interface.
Common functions for session management, authentication, and UI utilities.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from datetime import datetime
import re

from .api_client import APIClient, authenticate_user, logout_user


def initialize_session_state():
    """Initialize Streamlit session state with default values."""

    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'user' not in st.session_state:
        st.session_state.user = None

    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None

    # Chat state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None

    # UI state
    if 'page_config' not in st.session_state:
        st.session_state.page_config = {
            'theme': 'auto',
            'show_sql': True,
            'auto_execute': True,
            'max_results': 100
        }

    # Cache for API responses
    if 'api_cache' not in st.session_state:
        st.session_state.api_cache = {}


def handle_authentication() -> bool:
    """
    Handle user authentication flow.

    Returns:
        True if user is authenticated, False otherwise
    """

    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        return True

    # Show authentication UI
    return render_authentication_ui()


def render_authentication_ui() -> bool:
    """Render authentication UI and handle login/register."""

    st.title("🔐 ChatBI Login")
    st.markdown("Please log in to access the ChatBI platform.")

    # Create tabs for login and register
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    with tab1:
        return render_login_form()

    with tab2:
        return render_register_form()


def render_login_form() -> bool:
    """Render login form and handle authentication."""

    with st.form("login_form"):
        st.subheader("Welcome Back!")

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            submit_button = st.form_submit_button("🔑 Login", use_container_width=True, type="primary")

        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
                return False

            with st.spinner("Authenticating..."):
                success = asyncio.run(authenticate_user(username, password))

                if success:
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                    return True
                else:
                    st.error("Invalid username or password.")
                    return False

    # Demo credentials info
    with st.expander("🎯 Demo Credentials", expanded=False):
        st.info("""
        **Demo Account:**
        - Username: `demo`
        - Password: `demo123`

        **Admin Account:**
        - Username: `admin`
        - Password: `admin123`
        """)

    return False


def render_register_form() -> bool:
    """Render registration form."""

    with st.form("register_form"):
        st.subheader("Create New Account")

        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="Enter your email")
        full_name = st.text_input("Full Name", placeholder="Enter your full name (optional)")
        password = st.text_input("Password", type="password", placeholder="Choose a password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            submit_button = st.form_submit_button("📝 Register", use_container_width=True, type="primary")

        if submit_button:
            # Validation
            if not username or not email or not password:
                st.error("Please fill in all required fields.")
                return False

            if password != confirm_password:
                st.error("Passwords do not match.")
                return False

            if len(password) < 6:
                st.error("Password must be at least 6 characters long.")
                return False

            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                return False

            with st.spinner("Creating account..."):
                from .api_client import register_user
                success = asyncio.run(register_user(username, email, password, full_name))

                if success:
                    st.success("Account created successfully! Please log in.")
                    return False  # Don't auto-login, let user login manually
                else:
                    return False

    return False


def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        if isinstance(timestamp, str):
            # Handle various timestamp formats
            if 'T' in timestamp:
                # ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Try common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%H:%M:%S']:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return timestamp

            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return str(timestamp)
    except Exception:
        return str(timestamp)


def format_number(value: Any) -> str:
    """Format number for display."""
    try:
        if isinstance(value, (int, float)):
            if abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                return f"{value / 1_000:.1f}K"
            elif isinstance(value, float):
                return f"{value:.2f}"
            else:
                return f"{value:,}"
        return str(value)
    except Exception:
        return str(value)


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string with fallback."""
    try:
        import json
        return json.loads(json_str)
    except Exception:
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """Create download link for data."""
    import base64

    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


def validate_sql_query(query: str) -> Tuple[bool, str]:
    """Basic SQL query validation."""
    query = query.strip().upper()

    # Check if it's a SELECT statement
    if not query.startswith('SELECT'):
        return False, "Only SELECT statements are allowed"

    # Check for dangerous keywords
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in query:
            return False, f"Dangerous keyword '{keyword}' detected"

    return True, "Query is valid"


def display_error_message(error: Exception, context: str = ""):
    """Display user-friendly error message."""
    error_str = str(error).lower()

    if "connection" in error_str:
        st.error(f"🔌 Connection Error: Unable to connect to the server. {context}")
    elif "timeout" in error_str:
        st.error(f"⏱️ Timeout Error: The request took too long. {context}")
    elif "permission" in error_str or "unauthorized" in error_str:
        st.error(f"🔒 Permission Error: You don't have access to this resource. {context}")
    elif "not found" in error_str:
        st.error(f"🔍 Not Found: The requested resource was not found. {context}")
    else:
        st.error(f"❌ Error: {str(error)} {context}")


def show_loading_spinner(message: str = "Loading..."):
    """Show loading spinner with message."""
    return st.spinner(message)


def display_success_message(message: str):
    """Display success message with consistent styling."""
    st.success(f"✅ {message}")


def display_info_message(message: str):
    """Display info message with consistent styling."""
    st.info(f"ℹ️ {message}")


def display_warning_message(message: str):
    """Display warning message with consistent styling."""
    st.warning(f"⚠️ {message}")


def create_metric_card(title: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Create a metric card with consistent styling."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )


def render_code_block(code: str, language: str = "sql"):
    """Render code block with syntax highlighting."""
    st.code(code, language=language)


def create_expandable_section(title: str, content: str, expanded: bool = False):
    """Create expandable section."""
    with st.expander(title, expanded=expanded):
        st.write(content)


def get_user_settings() -> Dict[str, Any]:
    """Get user settings from session state."""
    return st.session_state.get('page_config', {
        'theme': 'auto',
        'show_sql': True,
        'auto_execute': True,
        'max_results': 100
    })


def update_user_settings(settings: Dict[str, Any]):
    """Update user settings in session state."""
    if 'page_config' not in st.session_state:
        st.session_state.page_config = {}

    st.session_state.page_config.update(settings)


def clear_cache():
    """Clear all cached data."""
    if 'api_cache' in st.session_state:
        st.session_state.api_cache = {}

    # Clear other cache keys
    cache_keys = [key for key in st.session_state.keys() if key.startswith('cache_')]
    for key in cache_keys:
        del st.session_state[key]


def get_theme_config():
    """Get theme configuration."""
    settings = get_user_settings()
    theme = settings.get('theme', 'auto')

    if theme == 'dark':
        return {
            'backgroundColor': '#0E1117',
            'secondaryBackgroundColor': '#262730',
            'textColor': '#FAFAFA'
        }
    elif theme == 'light':
        return {
            'backgroundColor': '#FFFFFF',
            'secondaryBackgroundColor': '#F0F2F6',
            'textColor': '#262730'
        }
    else:
        return {}  # Auto theme


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe downloads."""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    return filename


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate DataFrame for display."""
    if df is None:
        return False, "DataFrame is None"

    if df.empty:
        return False, "DataFrame is empty"

    if len(df.columns) == 0:
        return False, "DataFrame has no columns"

    # Check for reasonable size
    if len(df) > 50000:
        return False, f"DataFrame too large ({len(df)} rows). Consider filtering."

    return True, "DataFrame is valid"


def create_sidebar_info(title: str, content: Dict[str, Any]):
    """Create informational sidebar section."""
    with st.sidebar:
        st.subheader(title)
        for key, value in content.items():
            st.write(f"**{key}:** {value}")


def handle_navigation(page_name: str):
    """Handle page navigation."""
    if page_name == "overview":
        st.switch_page("pages/1_📊_Overview.py")
    elif page_name == "chat":
        st.switch_page("pages/2_💬_Chat.py")
    elif page_name == "home":
        st.switch_page("ui/app.py")


def get_sample_queries() -> List[str]:
    """Get sample queries for user guidance."""
    return [
        "Show me all customers from the last month",
        "What are the top 5 products by sales?",
        "Display revenue trends for this year",
        "How many orders were placed today?",
        "Which regions have the highest sales?",
        "Show me customer demographics",
        "What's the average order value?",
        "List all products with low inventory",
        "Show sales by category",
        "Find customers with no recent orders"
    ]


def create_help_text(section: str) -> str:
    """Create context-specific help text."""
    help_texts = {
        "chat": """
        **Chat Tips:**
        - Ask questions in plain English
        - Be specific about what data you want
        - Mention time periods when relevant
        - Use follow-up questions to drill down
        """,
        "overview": """
        **Overview Features:**
        - View system health and performance
        - Browse available data tables
        - Check recent activity
        - Monitor system metrics
        """,
        "data": """
        **Data Tips:**
        - Explore table schemas before querying
        - Use filters to narrow down results
        - Export data in various formats
        - Check data quality metrics
        """
    }

    return help_texts.get(section, "No help available for this section.")


def render_footer():
    """Render consistent footer across pages."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**ChatBI** - Intelligent Data Analytics")

    with col2:
        st.markdown("Made with ❤️ using Streamlit")

    with col3:
        st.markdown(f"Version {get_app_version()}")


def get_app_version() -> str:
    """Get application version."""
    from utils.config import settings
    return settings.app_version


def check_user_permissions(required_permission: str) -> bool:
    """Check if current user has required permission."""
    user = st.session_state.get('user')
    if not user:
        return False

    # Simple permission check - in production, this would be more sophisticated
    user_permissions = user.get('permissions', [])
    is_admin = user.get('is_admin', False)

    return is_admin or required_permission in user_permissions


def require_permission(permission: str):
    """Decorator/function to require specific permission."""
    if not check_user_permissions(permission):
        st.error(f"Permission denied. Required permission: {permission}")
        st.stop()


def log_user_action(action: str, details: Optional[Dict[str, Any]] = None):
    """Log user action for analytics."""
    # In production, this would send to analytics service
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user': st.session_state.get('user', {}).get('username', 'anonymous'),
        'action': action,
        'details': details or {}
    }

    # For now, just store in session state
    if 'user_actions' not in st.session_state:
        st.session_state.user_actions = []

    st.session_state.user_actions.append(log_entry)


def get_user_activity_summary() -> Dict[str, Any]:
    """Get summary of user activity."""
    actions = st.session_state.get('user_actions', [])

    return {
        'total_actions': len(actions),
        'last_action': actions[-1]['timestamp'] if actions else None,
        'action_counts': {}  # Would count action types
    }


def reset_session():
    """Reset session state (logout + clear cache)."""
    # Keep only essential keys
    keys_to_keep = ['page_config']

    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    # Reinitialize
    initialize_session_state()


def create_status_indicator(status: str) -> str:
    """Create colored status indicator."""
    status_colors = {
        'success': '🟢',
        'warning': '🟡',
        'error': '🔴',
        'info': '🔵',
        'pending': '⚪'
    }

    return status_colors.get(status.lower(), '⚪')


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def create_progress_bar(current: int, total: int, label: str = "") -> None:
    """Create progress bar with label."""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label} ({current}/{total})")


def render_data_quality_badge(quality_score: float) -> str:
    """Render data quality badge based on score."""
    if quality_score >= 0.9:
        return "🟢 Excellent"
    elif quality_score >= 0.7:
        return "🟡 Good"
    elif quality_score >= 0.5:
        return "🟠 Fair"
    else:
        return "🔴 Poor"