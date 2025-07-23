"""
Chat page for ChatBI platform.
Interactive chat interface for natural language data analysis.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.utils.api_client import APIClient
from ui.components.chat_interface import render_chat_interface
from ui.components.charts import render_chart_from_config
from ui.components.data_table import render_data_table
from utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Chat Interface - ChatBI",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        background: #f8fafc;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        margin-left: 2rem;
        position: relative;
    }

    .assistant-message {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        margin-right: 2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .sql-container {
        background: #1a202c;
        color: #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.875rem;
        overflow-x: auto;
    }

    .query-stats {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    }

    .suggestion-button {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
        display: inline-block;
    }

    .suggestion-button:hover {
        background: #e2e8f0;
        border-color: #94a3b8;
    }

    .thinking-indicator {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main chat page function."""
    try:
        st.title("💬 Chat with Your Data")
        st.markdown("Ask questions about your data in natural language and get instant insights.")

        # Initialize API client
        api_client = APIClient()

        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None

        # Sidebar for chat management
        with st.sidebar:
            render_chat_sidebar(api_client)

        # Main chat interface
        render_main_chat_area(api_client)

    except Exception as e:
        logger.error(f"Chat page error: {e}")
        st.error(f"Failed to load chat interface: {e}")


def render_chat_sidebar(api_client: APIClient):
    """Render chat sidebar with session management."""
    st.subheader("💬 Chat Sessions")

    # New chat button
    if st.button("🆕 New Chat", use_container_width=True, type="primary"):
        start_new_chat_session(api_client)

    st.markdown("---")

    # Load chat sessions
    try:
        sessions = asyncio.run(api_client.get_chat_sessions())

        if sessions:
            st.subheader("📝 Recent Chats")

            for session in sessions[:10]:  # Show last 10 sessions
                session_name = session.get('session_name', 'Unnamed Chat')
                session_id = session.get('id')
                created_at = session.get('created_at', '')

                # Truncate long session names
                display_name = session_name[:30] + "..." if len(session_name) > 30 else session_name

                if st.button(
                        f"📄 {display_name}",
                        key=f"session_{session_id}",
                        use_container_width=True,
                        help=f"Created: {created_at}"
                ):
                    load_chat_session(api_client, session_id)
        else:
            st.info("No previous chat sessions found.")

    except Exception as e:
        st.error(f"Failed to load chat sessions: {e}")

    st.markdown("---")

    # Quick help
    st.subheader("💡 Quick Tips")
    st.markdown("""
    **Getting better results:**
    - Be specific about what you want
    - Mention time periods if relevant
    - Ask follow-up questions
    - Use "show me" or "what are" to start

    **Example questions:**
    - "Show me sales by month"
    - "Which customers bought the most?"
    - "What's our revenue trend?"
    """)

    # Chat settings
    with st.expander("⚙️ Chat Settings"):
        auto_execute = st.checkbox("Auto-execute queries", value=True)
        show_sql = st.checkbox("Show generated SQL", value=True)
        max_results = st.slider("Max results", 10, 500, 100)

        st.session_state.chat_settings = {
            'auto_execute': auto_execute,
            'show_sql': show_sql,
            'max_results': max_results
        }


def render_main_chat_area(api_client: APIClient):
    """Render the main chat conversation area."""

    # Check for example query from home page
    if 'example_query' in st.session_state:
        example_query = st.session_state.example_query
        del st.session_state.example_query

        # Process the example query
        asyncio.run(process_user_message(api_client, example_query))
        st.rerun()

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                render_chat_message(message)
        else:
            # Welcome message for new chat
            st.markdown("""
            <div class="assistant-message">
                <h3>👋 Welcome to ChatBI!</h3>
                <p>I'm here to help you analyze your data. You can ask me questions like:</p>
                <ul>
                    <li>"Show me the top 10 customers by revenue"</li>
                    <li>"What were our sales trends last quarter?"</li>
                    <li>"How many orders did we process this month?"</li>
                </ul>
                <p>Just type your question below and I'll generate the SQL and show you the results!</p>
            </div>
            """, unsafe_allow_html=True)

    # Suggested questions for new users
    if not st.session_state.messages:
        st.subheader("🚀 Try these questions:")

        suggestions = [
            "Show me all tables in the database",
            "What's the total number of customers?",
            "Display sales data for the last 30 days",
            "Which products are selling the best?",
            "Show me user activity trends"
        ]

        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i % len(cols)]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    asyncio.run(process_user_message(api_client, suggestion))
                    st.rerun()

    # Chat input
    st.markdown("---")

    # Input form
    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., Show me revenue by month for this year",
                label_visibility="collapsed"
            )

        with col2:
            submit_button = st.form_submit_button("Send 🚀", use_container_width=True)

        if submit_button and user_input.strip():
            # Process the user message
            with st.spinner("🤔 Thinking..."):
                asyncio.run(process_user_message(api_client, user_input.strip()))
            st.rerun()


def render_chat_message(message):
    """Render an individual chat message."""
    message_type = message.get('type', 'user')
    content = message.get('content', '')
    timestamp = message.get('timestamp', '')

    if message_type == 'user':
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong> <small>({timestamp})</small><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

    elif message_type == 'assistant':
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 ChatBI Assistant</strong> <small>({timestamp})</small><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

        # Show SQL if available
        if 'sql_query' in message and message['sql_query']:
            st.markdown("**Generated SQL:**")
            st.markdown(f"""
            <div class="sql-container">
{message['sql_query']}
            </div>
            """, unsafe_allow_html=True)

        # Show execution stats
        if 'execution_stats' in message:
            stats = message['execution_stats']
            st.markdown(f"""
            <div class="query-stats">
                ⏱️ Execution time: {stats.get('execution_time_ms', 0)}ms | 
                📊 Results: {stats.get('row_count', 0)} rows | 
                ✅ Status: {stats.get('status', 'Unknown')}
            </div>
            """, unsafe_allow_html=True)

        # Show data table if available
        if 'data' in message and message['data']:
            st.markdown("**Results:**")
            df = pd.DataFrame(message['data'])
            render_data_table(df, max_rows=50)

        # Show chart if available
        if 'chart_config' in message and message['chart_config']:
            st.markdown("**Visualization:**")
            try:
                render_chart_from_config(message['chart_config'], message.get('data', []))
            except Exception as e:
                st.warning(f"Could not render chart: {e}")

        # Show follow-up suggestions
        if 'suggestions' in message and message['suggestions']:
            st.markdown("**Follow-up questions:**")
            cols = st.columns(len(message['suggestions']))
            for i, suggestion in enumerate(message['suggestions']):
                with cols[i]:
                    if st.button(suggestion, key=f"followup_{message.get('id', 0)}_{i}"):
                        # Add suggestion as new user message
                        api_client = APIClient()
                        asyncio.run(process_user_message(api_client, suggestion))
                        st.rerun()

    elif message_type == 'error':
        st.error(f"❌ Error: {content}")

    elif message_type == 'thinking':
        st.markdown(f"""
        <div class="thinking-indicator">
            🤔 {content}
        </div>
        """, unsafe_allow_html=True)


async def process_user_message(api_client: APIClient, user_input: str):
    """Process user message and get AI response."""
    try:
        # Add user message to chat
        user_message = {
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'id': len(st.session_state.messages)
        }
        st.session_state.messages.append(user_message)

        # Add thinking indicator
        thinking_message = {
            'type': 'thinking',
            'content': 'Analyzing your question and generating SQL...',
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'id': len(st.session_state.messages)
        }
        st.session_state.messages.append(thinking_message)

        # Process query with API
        query_response = await api_client.process_query(
            question=user_input,
            session_id=st.session_state.current_session_id,
            include_chart_suggestion=True
        )

        # Remove thinking indicator
        st.session_state.messages.pop()

        # Create assistant response
        assistant_message = {
            'type': 'assistant',
            'content': query_response.get('result_summary', 'Here are your results:'),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'id': len(st.session_state.messages),
            'sql_query': query_response.get('generated_sql', ''),
            'data': query_response.get('result_data', []),
            'execution_stats': {
                'execution_time_ms': query_response.get('execution_time_ms', 0),
                'row_count': len(query_response.get('result_data', [])),
                'status': query_response.get('execution_status', 'unknown')
            },
            'chart_config': query_response.get('chart_suggestion'),
            'suggestions': await api_client.get_follow_up_suggestions(
                user_input,
                query_response.get('result_data', [])
            )
        }

        # Handle errors
        if query_response.get('execution_status') == 'error':
            assistant_message['type'] = 'error'
            assistant_message['content'] = query_response.get('error_message', 'An error occurred')

        st.session_state.messages.append(assistant_message)

    except Exception as e:
        logger.error(f"Error processing user message: {e}")

        # Remove thinking indicator if present
        if st.session_state.messages and st.session_state.messages[-1]['type'] == 'thinking':
            st.session_state.messages.pop()

        # Add error message
        error_message = {
            'type': 'error',
            'content': f"Sorry, I encountered an error: {str(e)}",
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'id': len(st.session_state.messages)
        }
        st.session_state.messages.append(error_message)


def start_new_chat_session(api_client: APIClient):
    """Start a new chat session."""
    try:
        # Clear current messages
        st.session_state.messages = []

        # Create new session via API
        session_name = f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
        new_session = asyncio.run(api_client.create_chat_session(session_name))

        st.session_state.current_session_id = new_session.get('id')
        st.success(f"Started new chat session: {session_name}")

    except Exception as e:
        st.error(f"Failed to create new chat session: {e}")


def load_chat_session(api_client: APIClient, session_id: int):
    """Load an existing chat session."""
    try:
        # Load session messages
        session_data = asyncio.run(api_client.get_chat_session(session_id))

        # Convert API messages to UI format
        st.session_state.messages = []
        for msg in session_data.get('messages', []):
            ui_message = {
                'type': 'user' if msg['message_type'] == 'user' else 'assistant',
                'content': msg['content'],
                'timestamp': msg['created_at'][:8] if msg.get('created_at') else '',
                'id': msg['id']
            }

            # Add metadata for assistant messages
            if msg['message_type'] == 'assistant' and msg.get('metadata'):
                metadata = msg['metadata']
                ui_message.update({
                    'execution_stats': {
                        'execution_time_ms': metadata.get('execution_time_ms', 0),
                        'row_count': metadata.get('result_rows', 0),
                        'status': metadata.get('execution_status', 'unknown')
                    }
                })

            st.session_state.messages.append(ui_message)

        st.session_state.current_session_id = session_id
        st.success(f"Loaded chat session: {session_data.get('session_name', 'Unknown')}")

    except Exception as e:
        st.error(f"Failed to load chat session: {e}")


if __name__ == "__main__":
    main()