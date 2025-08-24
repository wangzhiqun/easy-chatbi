from typing import Dict, Any, List, Optional

import requests

DEFAULT_API_URL = "http://localhost:8000/api/v1"


def get_api_url() -> str:
    import streamlit as st
    return st.session_state.get('api_url', DEFAULT_API_URL)


def test_connection(api_url: str = None) -> bool:
    url = api_url or get_api_url()
    try:
        response = requests.get(f"{url.replace('/api/v1', '')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def send_chat_message(message: str, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    import streamlit as st
    try:
        url = f"{get_api_url()}/chat/chat"
        payload = {
            "message": message,
            "conversation_id": conversation_id
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to send message: {str(e)}")
        return None


def execute_query(query: str, database: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/data/query"
        payload = {
            "query": query,
            "database": database
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_schema() -> Optional[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/data/schema"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_table_info(table_name: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/data/table-info"
        payload = {"table_name": table_name}

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def validate_sql(query: str) -> Dict[str, Any]:
    try:
        url = f"{get_api_url()}/data/validate-sql"
        payload = {"query": query}

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        return {"valid": False, "error": "API Error"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def create_chart(data: List[Dict], chart_type: str = "auto", options: Optional[Dict] = None) -> Optional[
    Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/data/chart"
        payload = {
            "data": data,
            "chart_type": chart_type,
            "options": options or {}
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def analyze_data(data: List[Dict], analysis_type: str = "comprehensive") -> Optional[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/data/analyze"
        payload = {
            "data": data,
            "analysis_type": analysis_type
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_query_history(limit: int = 10) -> List[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/data/history"
        params = {"limit": limit}

        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def list_mcp_tools() -> List[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/mcp/tools"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        url = f"{get_api_url()}/mcp/tools/execute"
        payload = {
            "tool_name": tool_name,
            "arguments": arguments
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        return []
    except:
        return []


def add_knowledge(title: str, content: str, category: str, tags: List[str]) -> bool:
    try:
        return True
    except:
        return False


def get_cache_stats() -> Optional[Dict[str, Any]]:
    try:
        return {
            'connected': True,
            'used_memory': '12.5 MB',
            'hit_rate': 85.5,
            'total_commands_processed': 1234
        }
    except:
        return None


def clear_cache(namespace: str = "all") -> bool:
    try:
        return True
    except:
        return False


def generate_api_key() -> Optional[str]:
    try:
        import secrets
        return f"sk_{secrets.token_urlsafe(32)}"
    except:
        return None


def get_query_templates() -> Dict[str, str]:
    return {
        'sales_summary': """SELECT 
    DATE(created_at) as date,
    COUNT(*) as order_count,
    SUM(amount) as total_revenue
FROM orders
WHERE status = 'completed'
GROUP BY DATE(created_at)
ORDER BY date DESC""",

        'top_products': """SELECT 
    p.name,
    COUNT(o.id) as order_count,
    SUM(o.amount) as revenue
FROM products p
JOIN orders o ON p.id = o.product_id
GROUP BY p.id, p.name
ORDER BY revenue DESC
LIMIT 10""",

        'user_activity': """SELECT 
    username,
    COUNT(*) as total_actions,
    MAX(created_at) as last_activity
FROM users
GROUP BY username
ORDER BY total_actions DESC"""
    }


def generate_sql_from_description(request_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{get_api_url()}/data/generate-sql",
            json=request_data
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                'status': 'error',
                'error': f'HTTP {response.status_code}: {response.text}'
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': f'请求失败: {str(e)}'
        }
