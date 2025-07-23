"""
Chat interface components for ChatBI platform.
Reusable components for building chat interactions.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

from ..utils.api_client import APIClient
from .charts import render_chart_from_config
from .data_table import render_data_table


def render_chat_interface(
        messages: List[Dict[str, Any]],
        on_user_input: callable,
        placeholder_text: str = "Ask a question about your data...",
        show_suggestions: bool = True
):
    """
    Render a complete chat interface with message history and input.

    Args:
        messages: List of chat messages
        on_user_input: Callback function for user input
        placeholder_text: Placeholder for input field
        show_suggestions: Whether to show suggested questions
    """

    # Chat messages container
    chat_container = st.container()

    with chat_container:
        if messages:
            for message in messages:
                render_message_bubble(message)
        else:
            render_welcome_message()

    # Suggestions for new conversations
    if show_suggestions and not messages:
        render_suggested_questions(on_user_input)

    # Input area
    render_chat_input(on_user_input, placeholder_text)


def render_message_bubble(message: Dict[str, Any]):
    """
    Render an individual message bubble.

    Args:
        message: Message dictionary with type, content, timestamp, etc.
    """
    message_type = message.get('type', 'user')
    content = message.get('content', '')
    timestamp = message.get('timestamp', datetime.now().strftime('%H:%M:%S'))

    if message_type == 'user':
        st.markdown(f"""
        <div style="text-align: right; margin: 1rem 0;">
            <div style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.75rem 1rem; border-radius: 1rem 1rem 0.25rem 1rem; 
                        max-width: 70%; word-wrap: break-word;">
                <div style="font-size: 0.75rem; opacity: 0.8; margin-bottom: 0.25rem;">You • {timestamp}</div>
                <div>{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif message_type == 'assistant':
        render_assistant_message(message, timestamp)

    elif message_type == 'error':
        st.error(f"❌ {content}")

    elif message_type == 'system':
        st.info(f"ℹ️ {content}")


def render_assistant_message(message: Dict[str, Any], timestamp: str):
    """Render assistant message with all components."""

    # Main message bubble
    st.markdown(f"""
    <div style="text-align: left; margin: 1rem 0;">
        <div style="display: inline-block; background: white; border: 1px solid #e2e8f0;
                    padding: 0.75rem 1rem; border-radius: 1rem 1rem 1rem 0.25rem; 
                    max-width: 70%; word-wrap: break-word; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem;">
                🤖 ChatBI Assistant • {timestamp}
            </div>
            <div>{message.get('content', '')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SQL Query display
    if message.get('sql_query'):
        render_sql_display(message['sql_query'])

    # Execution statistics
    if message.get('execution_stats'):
        render_execution_stats(message['execution_stats'])

    # Data results
    if message.get('data'):
        render_query_results(message['data'])

    # Chart visualization
    if message.get('chart_config'):
        render_query_chart(message['chart_config'], message.get('data', []))

    # Follow-up suggestions
    if message.get('suggestions'):
        render_followup_suggestions(message['suggestions'], message.get('id'))


def render_sql_display(sql_query: str):
    """Render SQL query with syntax highlighting."""
    with st.expander("📝 Generated SQL Query", expanded=False):
        st.code(sql_query, language='sql')

        # Copy button (simulated)
        if st.button("📋 Copy SQL", key=f"copy_sql_{hash(sql_query)}"):
            st.success("SQL copied to clipboard! (Simulated)")


def render_execution_stats(stats: Dict[str, Any]):
    """Render query execution statistics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⏱️ Execution Time",
            f"{stats.get('execution_time_ms', 0)}ms"
        )

    with col2:
        st.metric(
            "📊 Results",
            f"{stats.get('row_count', 0)} rows"
        )

    with col3:
        status = stats.get('status', 'unknown')
        status_emoji = "✅" if status == 'success' else "❌"
        st.metric(
            "📈 Status",
            f"{status_emoji} {status.title()}"
        )


def render_query_results(data: List[Dict[str, Any]]):
    """Render query results in a table."""
    if not data:
        st.info("No data returned from query.")
        return

    st.markdown("### 📊 Query Results")

    try:
        df = pd.DataFrame(data)
        render_data_table(df, max_rows=100)

        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Download CSV", key=f"csv_{hash(str(data))}"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📊 Download Excel", key=f"excel_{hash(str(data))}"):
                # In a real implementation, create Excel file
                st.info("Excel download would be implemented here")

        with col3:
            if st.button("📋 Copy Data", key=f"copy_{hash(str(data))}"):
                st.success("Data copied to clipboard! (Simulated)")

    except Exception as e:
        st.error(f"Failed to display results: {e}")


def render_query_chart(chart_config: Dict[str, Any], data: List[Dict[str, Any]]):
    """Render chart visualization for query results."""
    if not data or not chart_config:
        return

    st.markdown("### 📈 Visualization")

    try:
        render_chart_from_config(chart_config, data)

        # Chart options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎨 Customize Chart", key=f"customize_{hash(str(chart_config))}"):
                render_chart_customization(chart_config, data)

        with col2:
            if st.button("💾 Save Chart", key=f"save_chart_{hash(str(chart_config))}"):
                st.success("Chart saved! (Simulated)")

    except Exception as e:
        st.warning(f"Could not render chart: {e}")


def render_followup_suggestions(suggestions: List[str], message_id: Optional[int] = None):
    """Render follow-up question suggestions."""
    if not suggestions:
        return

    st.markdown("### 💡 Follow-up Questions")

    # Display suggestions as clickable buttons
    cols = st.columns(min(len(suggestions), 3))
    for i, suggestion in enumerate(suggestions):
        with cols[i % len(cols)]:
            if st.button(
                    suggestion,
                    key=f"suggestion_{message_id}_{i}",
                    use_container_width=True
            ):
                # Return suggestion to be processed
                st.session_state.pending_suggestion = suggestion
                st.rerun()


def render_suggested_questions(on_user_input: callable):
    """Render suggested questions for new conversations."""
    st.markdown("### 🚀 Try asking:")

    suggestions = [
        "Show me all available tables",
        "What's the total revenue this month?",
        "Which are our top 5 customers?",
        "Display user activity trends",
        "How many orders were placed today?"
    ]

    # Create a grid of suggestion buttons
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(
                    f"💬 {suggestion}",
                    key=f"initial_suggestion_{i}",
                    use_container_width=True
            ):
                asyncio.run(on_user_input(suggestion))


def render_chat_input(on_user_input: callable, placeholder_text: str):
    """Render chat input form."""

    # Check for pending suggestions
    if 'pending_suggestion' in st.session_state:
        suggestion = st.session_state.pending_suggestion
        del st.session_state.pending_suggestion
        asyncio.run(on_user_input(suggestion))
        st.rerun()

    st.markdown("---")

    with st.form("chat_input", clear_on_submit=True):
        col1, col2, col3 = st.columns([6, 1, 1])

        with col1:
            user_input = st.text_input(
                "Message",
                placeholder=placeholder_text,
                label_visibility="collapsed"
            )

        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)

        with col3:
            clear = st.form_submit_button("Clear", use_container_width=True)

        if submit and user_input.strip():
            asyncio.run(on_user_input(user_input.strip()))
            st.rerun()

        if clear:
            st.session_state.messages = []
            st.rerun()


def render_welcome_message():
    """Render welcome message for new chat sessions."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                padding: 2rem; border-radius: 1rem; border: 1px solid #0ea5e9; margin: 1rem 0;">
        <h2 style="color: #0c4a6e; margin-bottom: 1rem;">👋 Welcome to ChatBI!</h2>
        <p style="color: #164e63; margin-bottom: 1rem;">
            I'm your AI data analyst. I can help you explore your data, generate insights, 
            and create visualizations using natural language.
        </p>
        <h4 style="color: #0c4a6e; margin-bottom: 0.5rem;">What can I help you with?</h4>
        <ul style="color: #164e63;">
            <li>📊 Generate charts and visualizations</li>
            <li>🔍 Query your data with natural language</li>
            <li>📈 Analyze trends and patterns</li>
            <li>📋 Create reports and summaries</li>
            <li>💡 Discover insights in your data</li>
        </ul>
        <p style="color: #164e63; margin-top: 1rem; font-style: italic;">
            Just type your question below and I'll take care of the rest!
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_chat_history_sidebar(messages: List[Dict[str, Any]]):
    """Render chat history in sidebar."""
    st.sidebar.subheader("💬 Chat History")

    if not messages:
        st.sidebar.info("No messages yet")
        return

    # Show recent messages in sidebar
    for i, message in enumerate(reversed(messages[-10:])):  # Last 10 messages
        if message['type'] == 'user':
            content = message['content'][:50] + "..." if len(message['content']) > 50 else message['content']

            if st.sidebar.button(
                    f"👤 {content}",
                    key=f"history_{i}",
                    use_container_width=True
            ):
                # Jump to message (implementation would scroll to message)
                st.info(f"Jumped to: {content}")


def render_typing_indicator():
    """Render typing indicator for AI responses."""
    st.markdown("""
    <div style="background: #f1f5f9; padding: 1rem; border-radius: 0.5rem; 
                margin: 0.5rem 0; text-align: center; border: 1px solid #cbd5e1;">
        <div style="display: inline-flex; align-items: center; gap: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #64748b; 
                        animation: pulse 1.5s ease-in-out infinite;"></div>
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #64748b; 
                        animation: pulse 1.5s ease-in-out infinite 0.3s;"></div>
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #64748b; 
                        animation: pulse 1.5s ease-in-out infinite 0.6s;"></div>
            <span style="margin-left: 0.5rem; color: #64748b;">ChatBI is thinking...</span>
        </div>
    </div>

    <style>
    @keyframes pulse {
        0%, 60%, 100% { opacity: 0.3; }
        30% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_chart_customization(chart_config: Dict[str, Any], data: List[Dict[str, Any]]):
    """Render chart customization interface."""
    with st.expander("🎨 Customize Chart", expanded=True):

        # Chart type selection
        chart_types = ['bar', 'line', 'pie', 'scatter', 'area', 'histogram']
        current_type = chart_config.get('chart_type', 'bar')

        new_chart_type = st.selectbox(
            "Chart Type",
            chart_types,
            index=chart_types.index(current_type) if current_type in chart_types else 0
        )

        # Column selection
        if data:
            columns = list(data[0].keys())

            x_axis = st.selectbox(
                "X-Axis",
                columns,
                index=columns.index(chart_config.get('x_axis', columns[0])) if chart_config.get(
                    'x_axis') in columns else 0
            )

            y_axis = st.selectbox(
                "Y-Axis",
                columns,
                index=columns.index(
                    chart_config.get('y_axis', columns[1] if len(columns) > 1 else columns[0])) if chart_config.get(
                    'y_axis') in columns else 1 if len(columns) > 1 else 0
            )

            # Chart title
            title = st.text_input(
                "Chart Title",
                value=chart_config.get('title', 'Data Visualization')
            )

            # Apply changes
            if st.button("Apply Changes"):
                new_config = {
                    'chart_type': new_chart_type,
                    'x_axis': x_axis,
                    'y_axis': y_axis,
                    'title': title
                }

                try:
                    render_chart_from_config(new_config, data)
                    st.success("Chart updated!")
                except Exception as e:
                    st.error(f"Failed to update chart: {e}")


def render_message_actions(message: Dict[str, Any]):
    """Render action buttons for messages."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👍", key=f"like_{message.get('id')}"):
            st.success("Feedback recorded!")

    with col2:
        if st.button("👎", key=f"dislike_{message.get('id')}"):
            st.info("Thanks for the feedback!")

    with col3:
        if st.button("🔄", key=f"retry_{message.get('id')}"):
            st.info("Retrying query...")


def format_message_timestamp(timestamp):
    """Format timestamp for display."""
    try:
        if isinstance(timestamp, str):
            # Try to parse various timestamp formats
            for fmt in ['%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.strftime('%H:%M:%S')
                except ValueError:
                    continue
        return timestamp
    except:
        return datetime.now().strftime('%H:%M:%S')