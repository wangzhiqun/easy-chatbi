import os
import sys
from datetime import datetime
from typing import Dict, Any

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.utils import api_client


def render_chat_interface():
    st.title("💬 AI Chat Assistant")
    st.markdown("Ask questions about your data in natural language")

    initialize_chat_state()

    with st.sidebar:
        render_chat_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        render_chat_messages()
        render_chat_input()

    with col2:
        render_quick_actions()


def initialize_chat_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False


def render_chat_sidebar():
    st.subheader("Chat Options")

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    if st.button("💾 Save Chat", use_container_width=True):
        save_conversation()

    if st.button("📥 Load Chat", use_container_width=True):
        load_conversation()

    st.markdown("---")

    st.subheader("Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses"
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=2000,
        step=100,
        help="Maximum response length"
    )

    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens


def render_chat_messages():
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            render_message(message)


def render_message(message: Dict[str, Any]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "metadata" in message and message["metadata"]:
            render_message_metadata(message["metadata"])

        if "timestamp" in message:
            st.caption(f"🕐 {message['timestamp']}")


def render_message_metadata(metadata: Dict[str, Any]):
    with st.expander("📋 Details"):
        if "sql" in metadata:
            st.code(metadata["sql"], language="sql")

        if "result" in metadata and metadata["result"]:
            result = metadata["result"]
            if "data" in result and result["data"]:
                import pandas as pd
                df = pd.DataFrame(result["data"])
                st.dataframe(df, use_container_width=True)

        if "chart_config" in metadata:
            st.json(metadata["chart_config"])

        if "analysis_results" in metadata:
            st.json(metadata["analysis_results"])


def render_chat_input():
    if prompt := st.chat_input("Ask me anything about your data..."):
        handle_user_message(prompt)


def handle_user_message(prompt: str):
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M")
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = api_client.send_chat_message(
                prompt,
                st.session_state.conversation_id
            )

            if response:
                st.session_state.conversation_id = response.get('conversation_id')

                content = response.get('response', 'Sorry, I could not process your request.')
                metadata = response.get('metadata')

                st.markdown(content)

                if metadata:
                    render_message_metadata(metadata)

                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "metadata": metadata,
                    "timestamp": datetime.now().strftime("%H:%M")
                }
                st.session_state.messages.append(assistant_message)
            else:
                st.error("Failed to get response from AI")


def render_quick_actions():
    st.subheader("Quick Actions")

    quick_prompts = [
        "📊 Show database schema",
        "💰 Calculate total revenue",
        "📈 Show sales trend",
        "🔝 Top selling products",
        "👥 User activity report",
        "🔍 Find anomalies",
        "📉 Analyze patterns",
        "🎯 Performance metrics"
    ]

    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            clean_prompt = prompt.split(' ', 1)[1] if ' ' in prompt else prompt
            handle_user_message(clean_prompt)


def save_conversation():
    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"

        import json
        conversation_data = {
            "conversation_id": st.session_state.conversation_id,
            "messages": st.session_state.messages,
            "timestamp": timestamp
        }

        json_str = json.dumps(conversation_data, indent=2)

        st.download_button(
            label="💾 Download Conversation",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )

        st.success(f"Conversation saved as {filename}")
    else:
        st.warning("No messages to save")


def load_conversation():
    uploaded_file = st.file_uploader(
        "Choose a conversation file",
        type=['json'],
        key="conversation_uploader"
    )

    if uploaded_file is not None:
        import json

        try:
            conversation_data = json.load(uploaded_file)

            st.session_state.messages = conversation_data.get("messages", [])
            st.session_state.conversation_id = conversation_data.get("conversation_id")

            st.success("Conversation loaded successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load conversation: {str(e)}")


def render_typing_indicator():
    typing_placeholder = st.empty()
    with typing_placeholder.container():
        st.markdown("🤖 _AI is typing..._")
    return typing_placeholder
