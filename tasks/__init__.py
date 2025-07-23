"""
Tasks module for ChatBI platform.
Handles asynchronous tasks using Celery for AI analysis and data processing.
"""

from .celery_app import celery_app
from .ai_tasks import (
    generate_sql_async,
    analyze_data_async,
    generate_chart_async,
    process_chat_query_async
)
from .data_tasks import (
    import_data_async,
    process_csv_async,
    clean_data_async,
    validate_data_async,
    export_data_async
)

__all__ = [
    'celery_app',
    'generate_sql_async',
    'analyze_data_async',
    'generate_chart_async',
    'process_chat_query_async',
    'import_data_async',
    'process_csv_async',
    'clean_data_async',
    'validate_data_async',
    'export_data_async'
]