"""
API Client for ChatBI Streamlit interface.
Handles all communication with the FastAPI backend.
"""

import httpx
import streamlit as st
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime

from utils.config import settings


class APIClient:
    """
    Client for communicating with ChatBI FastAPI backend.
    Handles authentication, request/response processing, and error handling.
    """

    def __init__(self):
        """Initialize API client with configuration."""
        self.base_url = f"http://{settings.api_host}:{settings.api_port}"
        self.timeout = 30.0

        # Get auth token from session state
        self.auth_token = st.session_state.get('auth_token')

        # Default headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=self.headers, params=params)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=self.headers, json=data, params=params)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=self.headers, json=data, params=params)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=self.headers, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check for HTTP errors
                response.raise_for_status()

                # Return JSON response
                return response.json()

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            st.error(f"API Error: {error_msg}")
            raise Exception(f"API request failed: {error_msg}")

        except httpx.TimeoutException:
            st.error("Request timed out. Please try again.")
            raise Exception("API request timed out")

        except httpx.RequestError as e:
            st.error(f"Connection error: {str(e)}")
            raise Exception(f"Failed to connect to API: {str(e)}")

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            raise

    # Authentication methods

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Login user and get authentication token.

        Args:
            username: Username
            password: Password

        Returns:
            Login response with token
        """
        data = {"username": username, "password": password}
        response = await self._make_request("POST", "/api/auth/login", data=data)

        # Store token in session state
        if response.get('access_token'):
            st.session_state.auth_token = response['access_token']
            self.auth_token = response['access_token']
            self.headers["Authorization"] = f"Bearer {self.auth_token}"

        return response

    async def register(self, username: str, email: str, password: str, full_name: str = "") -> Dict[str, Any]:
        """
        Register new user.

        Args:
            username: Desired username
            email: User email
            password: Password
            full_name: Optional full name

        Returns:
            Registration response
        """
        data = {
            "username": username,
            "email": email,
            "password": password,
            "full_name": full_name
        }
        return await self._make_request("POST", "/api/auth/register", data=data)

    async def get_current_user(self) -> Dict[str, Any]:
        """Get current user information."""
        return await self._make_request("GET", "/api/auth/me")

    async def logout(self) -> Dict[str, Any]:
        """Logout current user."""
        response = await self._make_request("POST", "/api/auth/logout")

        # Clear token from session state
        if 'auth_token' in st.session_state:
            del st.session_state.auth_token
        self.auth_token = None
        if "Authorization" in self.headers:
            del self.headers["Authorization"]

        return response

    # Chat methods

    async def get_chat_sessions(self, skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's chat sessions."""
        params = {"skip": skip, "limit": limit}
        return await self._make_request("GET", "/api/chat/sessions", params=params)

    async def create_chat_session(self, session_name: str) -> Dict[str, Any]:
        """Create new chat session."""
        data = {"session_name": session_name}
        return await self._make_request("POST", "/api/chat/sessions", data=data)

    async def get_chat_session(self, session_id: int) -> Dict[str, Any]:
        """Get specific chat session with messages."""
        return await self._make_request("GET", f"/api/chat/sessions/{session_id}")

    async def update_chat_session(self, session_id: int, session_name: str) -> Dict[str, Any]:
        """Update chat session."""
        data = {"session_name": session_name}
        return await self._make_request("PUT", f"/api/chat/sessions/{session_id}", data=data)

    async def delete_chat_session(self, session_id: int) -> Dict[str, Any]:
        """Delete chat session."""
        return await self._make_request("DELETE", f"/api/chat/sessions/{session_id}")

    async def process_query(
            self,
            question: str,
            session_id: Optional[int] = None,
            include_chart_suggestion: bool = True
    ) -> Dict[str, Any]:
        """
        Process natural language query.

        Args:
            question: Natural language question
            session_id: Optional session ID
            include_chart_suggestion: Whether to include chart suggestions

        Returns:
            Query response with SQL, data, and analysis
        """
        data = {
            "question": question,
            "session_id": session_id,
            "include_chart_suggestion": include_chart_suggestion
        }
        return await self._make_request("POST", "/api/chat/query", data=data)

    async def chat_with_ai(
            self,
            question: str,
            session_id: Optional[int] = None,
            include_chart_suggestion: bool = True
    ) -> Dict[str, Any]:
        """
        Chat with AI (creates session if needed).

        Args:
            question: Natural language question
            session_id: Optional session ID
            include_chart_suggestion: Whether to include chart suggestions

        Returns:
            Chat response
        """
        data = {
            "question": question,
            "session_id": session_id,
            "include_chart_suggestion": include_chart_suggestion
        }
        return await self._make_request("POST", "/api/chat/chat", data=data)

    async def get_session_messages(
            self,
            session_id: int,
            skip: int = 0,
            limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get messages from a chat session."""
        params = {"skip": skip, "limit": limit}
        return await self._make_request("GET", f"/api/chat/sessions/{session_id}/messages", params=params)

    # Data methods

    async def get_tables(self) -> List[str]:
        """Get list of available database tables."""
        return await self._make_request("GET", "/api/data/tables")

    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema for a specific table."""
        return await self._make_request("GET", f"/api/data/tables/{table_name}/schema")

    async def get_sample_data(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from a table."""
        params = {"limit": limit}
        response = await self._make_request("GET", f"/api/data/tables/{table_name}/sample", params=params)
        return response.get('sample_data', [])

    async def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query directly."""
        data = {"sql_query": sql_query}
        return await self._make_request("POST", "/api/data/execute-sql", data=data)

    async def get_query_history(
            self,
            skip: int = 0,
            limit: int = 20,
            status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user's query history."""
        params = {"skip": skip, "limit": limit}
        if status_filter:
            params["status_filter"] = status_filter
        return await self._make_request("GET", "/api/data/query-history", params=params)

    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics and insights."""
        return await self._make_request("GET", "/api/data/statistics")

    async def create_visualization(
            self,
            chart_config: Dict[str, Any],
            sql_query: str
    ) -> Dict[str, Any]:
        """Create visualization from SQL query."""
        data = {
            "chart_config": chart_config,
            "sql_query": sql_query
        }
        return await self._make_request("POST", "/api/data/visualize", data=data)

    async def export_table_data(
            self,
            table_name: str,
            format: str = "csv",
            limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Export table data in specified format."""
        params = {"format": format}
        if limit:
            params["limit"] = limit
        return await self._make_request("GET", f"/api/data/export/{table_name}", params=params)

    # System methods

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        return await self._make_request("GET", "/health")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return await self._make_request("GET", "/metrics")

    # Utility methods

    async def get_follow_up_suggestions(
            self,
            question: str,
            results: List[Dict[str, Any]]
    ) -> List[str]:
        """Get follow-up question suggestions."""
        # This would be implemented as an API endpoint
        # For now, return some generic suggestions
        suggestions = []

        if results:
            if len(results) > 1:
                suggestions.append("Show me the top 5 results")
                suggestions.append("Can you break this down by category?")

            # Check for date columns
            if any('date' in str(key).lower() for key in results[0].keys()):
                suggestions.append("Show this data over time")

            # Check for numeric columns
            numeric_cols = [k for k, v in results[0].items() if isinstance(v, (int, float))]
            if numeric_cols:
                suggestions.append(f"What's the average {numeric_cols[0]}?")

        return suggestions[:3]  # Return top 3 suggestions

    async def get_recent_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent queries for the current user."""
        try:
            history = await self.get_query_history(limit=limit)
            return history
        except Exception as e:
            # Return empty list if we can't get history
            return []

    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.auth_token is not None

    async def validate_connection(self) -> bool:
        """Validate connection to the API."""
        try:
            await self.get_health_status()
            return True
        except Exception:
            return False


# Singleton instance for use throughout the app
_api_client = None


def get_api_client() -> APIClient:
    """Get singleton API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = APIClient()
    return _api_client


# Utility functions for common API operations

async def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user and store session data."""
    try:
        client = get_api_client()
        response = await client.login(username, password)

        if response.get('access_token'):
            # Get user info
            user_info = await client.get_current_user()
            st.session_state.user = user_info
            st.session_state.authenticated = True
            return True

        return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False


async def register_user(username: str, email: str, password: str, full_name: str = "") -> bool:
    """Register new user."""
    try:
        client = get_api_client()
        response = await client.register(username, email, password, full_name)
        return True
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False


async def logout_user():
    """Logout user and clear session data."""
    try:
        client = get_api_client()
        await client.logout()
    except Exception as e:
        st.warning(f"Logout error: {str(e)}")
    finally:
        # Clear session state regardless of API success
        for key in ['user', 'authenticated', 'auth_token']:
            if key in st.session_state:
                del st.session_state[key]


def handle_api_error(error: Exception, context: str = ""):
    """Handle API errors with user-friendly messages."""
    error_msg = str(error)

    if "401" in error_msg or "authentication" in error_msg.lower():
        st.error("Authentication failed. Please log in again.")
        # Clear authentication state
        for key in ['user', 'authenticated', 'auth_token']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    elif "403" in error_msg or "permission" in error_msg.lower():
        st.error("You don't have permission to perform this action.")

    elif "404" in error_msg:
        st.error(f"Resource not found. {context}")

    elif "timeout" in error_msg.lower():
        st.error("Request timed out. Please try again.")

    elif "connection" in error_msg.lower():
        st.error("Unable to connect to the server. Please check your connection.")

    else:
        st.error(f"An error occurred: {error_msg}")


async def test_api_connection() -> bool:
    """Test API connection and return status."""
    try:
        client = get_api_client()
        return await client.validate_connection()
    except Exception:
        return False


def format_api_timestamp(timestamp_str: str) -> str:
    """Format API timestamp for display."""
    try:
        # Try to parse ISO format
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return timestamp_str


def cache_api_response(key: str, data: Any, ttl: int = 300):
    """Cache API response in session state with TTL."""
    st.session_state[f"cache_{key}"] = {
        'data': data,
        'timestamp': datetime.now().timestamp(),
        'ttl': ttl
    }


def get_cached_response(key: str) -> Optional[Any]:
    """Get cached API response if still valid."""
    cache_key = f"cache_{key}"

    if cache_key in st.session_state:
        cache_data = st.session_state[cache_key]
        cache_age = datetime.now().timestamp() - cache_data['timestamp']

        if cache_age < cache_data['ttl']:
            return cache_data['data']
        else:
            # Remove expired cache
            del st.session_state[cache_key]

    return None


def clear_api_cache():
    """Clear all cached API responses."""
    cache_keys = [key for key in st.session_state.keys() if key.startswith('cache_')]
    for key in cache_keys:
        del st.session_state[key]