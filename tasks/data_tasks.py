"""
Data Tasks for ChatBI platform.
Handles asynchronous data processing, maintenance, and export operations.
"""

import json
import csv
import io
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from .celery_app import app
from services.data_service import DataService
from services.cache_service import CacheService
from api.database import get_table_names, get_table_schema
from utils.logger import get_logger
from utils.exceptions import DataProcessingException, DatabaseException

logger = get_logger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=60)
async def export_data(self, table_name: str, export_format: str, filters: Optional[Dict[str, Any]] = None,
                user_id: Optional[int] = None, max_rows: int = 50000):
    """
    Asynchronously export data from a table in specified format.

    Args:
        table_name: Name of table to export
        export_format: Format for export (csv, json, excel)
        filters: Optional filters to apply
        user_id: ID of user requesting export
        max_rows: Maximum number of rows to export

    Returns:
        Export result with file information
    """
    try:
        logger.info(f"Starting data export for table {table_name} in {export_format} format")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing data service', 'progress': 10}
        )

        # Initialize data service
        data_service = DataService()

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Retrieving data', 'progress': 30}
        )

        # Build query parameters
        where_clause = None
        if filters:
            where_conditions = []
            for field, value in filters.items():
                if isinstance(value, str):
                    where_conditions.append(f"{field} = '{value}'")
                else:
                    where_conditions.append(f"{field} = {value}")
            where_clause = " AND ".join(where_conditions)

        # Get data
        data = await data_service.get_table_data(
            table_name=table_name,
            where_clause=where_clause,
            limit=max_rows
        )

        self.update_state(
            state='PROCESSING',
            meta={'status': f'Exporting {len(data)} records', 'progress': 70}
        )

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{table_name}_export_{timestamp}.{export_format}"

        # Export data based on format
        if export_format.lower() == 'csv':
            export_content = _export_to_csv_content(data)
            content_type = 'text/csv'
        elif export_format.lower() == 'json':
            export_content = _export_to_json_content(data)
            content_type = 'application/json'
        elif export_format.lower() == 'excel':
            export_content = _export_to_excel_content(data, table_name)
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            raise DataProcessingException(f"Unsupported export format: {export_format}")

        # In a real implementation, you would save the file to a storage service
        # For now, we'll return metadata about the export

        result = {
            'success': True,
            'filename': filename,
            'format': export_format,
            'record_count': len(data),
            'file_size_bytes': len(export_content) if isinstance(export_content, (str, bytes)) else 0,
            'content_type': content_type,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat(),
            'download_url': f"/api/data/downloads/{filename}",  # Placeholder URL
            'task_info': {
                'task_id': self.request.id,
                'user_id': user_id,
                'table_name': table_name
            }
        }

        logger.info(f"Data export completed: {filename} ({len(data)} records)")
        return result

    except Exception as exc:
        logger.error(f"Data export failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))

        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=2)
async def refresh_cache(self, cache_keys: Optional[List[str]] = None, cache_pattern: Optional[str] = None):
    """
    Refresh cached data for improved performance.

    Args:
        cache_keys: Specific cache keys to refresh
        cache_pattern: Pattern to match cache keys for refresh

    Returns:
        Cache refresh results
    """
    try:
        logger.info("Starting cache refresh operation")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing cache service', 'progress': 10}
        )

        cache_service = CacheService()
        refreshed_count = 0

        if cache_keys:
            self.update_state(
                state='PROCESSING',
                meta={'status': f'Refreshing {len(cache_keys)} specific keys', 'progress': 30}
            )

            for key in cache_keys:
                try:
                    # Delete existing cache entry to force refresh
                    await cache_service.delete(key)
                    refreshed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to refresh cache key {key}: {e}")

        elif cache_pattern:
            self.update_state(
                state='PROCESSING',
                meta={'status': f'Refreshing keys matching pattern: {cache_pattern}', 'progress': 30}
            )

            # Invalidate cache entries matching pattern
            refreshed_count = await cache_service.invalidate_pattern(cache_pattern)

        else:
            # Refresh table metadata cache
            self.update_state(
                state='PROCESSING',
                meta={'status': 'Refreshing table metadata cache', 'progress': 30}
            )

            try:
                table_names = get_table_names()

                for i, table_name in enumerate(table_names):
                    try:
                        # Refresh table metadata
                        cache_key = f"metadata_{table_name}"
                        await cache_service.delete(cache_key)

                        # Update progress
                        progress = 30 + int((i / len(table_names)) * 50)
                        self.update_state(
                            state='PROCESSING',
                            meta={'status': f'Refreshing metadata for {table_name}', 'progress': progress}
                        )

                        refreshed_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to refresh metadata for {table_name}: {e}")

            except Exception as e:
                logger.error(f"Failed to get table names for cache refresh: {e}")

        result = {
            'success': True,
            'refreshed_count': refreshed_count,
            'refresh_type': 'specific_keys' if cache_keys else 'pattern' if cache_pattern else 'metadata',
            'completed_at': datetime.now().isoformat(),
            'task_info': {
                'task_id': self.request.id
            }
        }

        logger.info(f"Cache refresh completed: {refreshed_count} entries refreshed")
        return result

    except Exception as exc:
        logger.error(f"Cache refresh failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=1)
async def cleanup_expired_cache(self):
    """
    Clean up expired cache entries and optimize cache performance.

    Returns:
        Cleanup operation results
    """
    try:
        logger.info("Starting cache cleanup operation")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing cache usage', 'progress': 20}
        )

        cache_service = CacheService()

        # Get cache info before cleanup
        cache_info_before = await cache_service.get_cache_info()

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Cleaning expired entries', 'progress': 60}
        )

        # In a real implementation, this would:
        # 1. Scan for expired keys
        # 2. Remove stale cache entries
        # 3. Optimize memory usage
        # 4. Generate cleanup report

        # Simulate cleanup process
        cleanup_count = 0

        # Get cache statistics after cleanup
        cache_info_after = await cache_service.get_cache_info()

        result = {
            'success': True,
            'cleaned_entries': cleanup_count,
            'cache_stats_before': cache_info_before.get('stats', {}),
            'cache_stats_after': cache_info_after.get('stats', {}),
            'completed_at': datetime.now().isoformat(),
            'task_info': {
                'task_id': self.request.id
            }
        }

        logger.info(f"Cache cleanup completed: {cleanup_count} entries cleaned")
        return result

    except Exception as exc:
        logger.error(f"Cache cleanup failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=2)
async def update_table_statistics(self, table_names: Optional[List[str]] = None):
    """
    Update table statistics for query optimization.

    Args:
        table_names: Specific tables to update, or None for all tables

    Returns:
        Statistics update results
    """
    try:
        logger.info("Starting table statistics update")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Getting table list', 'progress': 10}
        )

        # Get tables to update
        if table_names is None:
            table_names = get_table_names()

        data_service = DataService()
        updated_tables = []
        failed_tables = []

        total_tables = len(table_names)

        for i, table_name in enumerate(table_names):
            try:
                progress = 10 + int((i / total_tables) * 80)
                self.update_state(
                    state='PROCESSING',
                    meta={'status': f'Updating statistics for {table_name}', 'progress': progress}
                )

                # Get table summary and cache it
                summary = await data_service.get_table_summary(table_name)
                await data_service.cache_table_metadata(table_name, force_refresh=True)

                updated_tables.append({
                    'table_name': table_name,
                    'row_count': summary.get('row_count', 0),
                    'column_count': summary.get('column_count', 0),
                    'updated_at': datetime.now().isoformat()
                })

            except Exception as e:
                logger.warning(f"Failed to update statistics for {table_name}: {e}")
                failed_tables.append({
                    'table_name': table_name,
                    'error': str(e)
                })

        result = {
            'success': True,
            'total_tables': total_tables,
            'updated_tables': len(updated_tables),
            'failed_tables': len(failed_tables),
            'table_details': updated_tables,
            'failures': failed_tables,
            'completed_at': datetime.now().isoformat(),
            'task_info': {
                'task_id': self.request.id
            }
        }

        logger.info(f"Table statistics update completed: {len(updated_tables)}/{total_tables} successful")
        return result

    except Exception as exc:
        logger.error(f"Table statistics update failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True)
def generate_usage_report(self, report_period_days: int = 7):
    """
    Generate usage analytics and performance reports.

    Args:
        report_period_days: Number of days to include in report

    Returns:
        Usage report data
    """
    try:
        logger.info(f"Generating usage report for {report_period_days} days")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Collecting usage data', 'progress': 20}
        )

        # In a real implementation, this would:
        # 1. Query audit logs for usage statistics
        # 2. Analyze query patterns and performance
        # 3. Generate user activity reports
        # 4. Create performance metrics
        # 5. Identify optimization opportunities

        # Simulate report generation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=report_period_days)

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing query patterns', 'progress': 50}
        )

        # Mock report data
        report_data = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': report_period_days
            },
            'summary': {
                'total_queries': 1250,  # Mock data
                'unique_users': 45,
                'avg_query_time_ms': 1850,
                'successful_queries': 1180,
                'failed_queries': 70
            },
            'top_queries': [
                {'query_pattern': 'SELECT COUNT(*) FROM sales', 'frequency': 89},
                {'query_pattern': 'SELECT * FROM customers WHERE...', 'frequency': 67},
                {'query_pattern': 'SELECT SUM(amount) FROM orders', 'frequency': 54}
            ],
            'performance_metrics': {
                'avg_response_time': 1.85,
                'cache_hit_rate': 78.5,
                'error_rate': 5.6
            },
            'user_activity': {
                'most_active_users': [
                    {'user_id': 1, 'query_count': 156},
                    {'user_id': 3, 'query_count': 134},
                    {'user_id': 7, 'query_count': 98}
                ],
                'peak_usage_hours': [9, 10, 14, 15, 16]
            },
            'recommendations': [
                'Consider optimizing queries with response time > 5 seconds',
                'Add indexes to frequently queried columns',
                'Implement query result caching for common patterns'
            ]
        }

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Finalizing report', 'progress': 90}
        )

        report_data['generated_at'] = datetime.now().isoformat()
        report_data['task_info'] = {
            'task_id': self.request.id,
            'report_type': 'usage_analytics'
        }

        logger.info("Usage report generation completed")
        return report_data

    except Exception as exc:
        logger.error(f"Usage report generation failed: {exc}")
        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


@app.task(bind=True, max_retries=2)
def backup_query_history(self, days_to_backup: int = 30, backup_format: str = 'json'):
    """
    Backup query history and audit logs.

    Args:
        days_to_backup: Number of days of history to backup
        backup_format: Format for backup (json, csv)

    Returns:
        Backup operation results
    """
    try:
        logger.info(f"Starting query history backup for {days_to_backup} days")

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Preparing backup data', 'progress': 20}
        )

        cutoff_date = datetime.now() - timedelta(days=days_to_backup)

        # In a real implementation, this would:
        # 1. Query the database for historical data
        # 2. Export query logs, audit trails, and user activities
        # 3. Compress and store backup files
        # 4. Clean up old backup files

        self.update_state(
            state='PROCESSING',
            meta={'status': 'Creating backup archive', 'progress': 70}
        )

        # Simulate backup process
        backup_filename = f"chatbi_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{backup_format}"

        result = {
            'success': True,
            'backup_filename': backup_filename,
            'backup_format': backup_format,
            'days_backed_up': days_to_backup,
            'cutoff_date': cutoff_date.isoformat(),
            'backup_size_mb': 45.7,  # Mock size
            'records_backed_up': 2847,  # Mock count
            'backup_location': f"/backups/{backup_filename}",
            'created_at': datetime.now().isoformat(),
            'task_info': {
                'task_id': self.request.id
            }
        }

        logger.info(f"Query history backup completed: {backup_filename}")
        return result

    except Exception as exc:
        logger.error(f"Query history backup failed: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            'success': False,
            'error': str(exc),
            'task_info': {
                'task_id': self.request.id,
                'failed_at': datetime.now().isoformat()
            }
        }


# Helper functions for data export

def _export_to_csv_content(data: List[Dict[str, Any]]) -> str:
    """Convert data to CSV format."""
    if not data:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def _export_to_json_content(data: List[Dict[str, Any]]) -> str:
    """Convert data to JSON format."""
    return json.dumps(data, indent=2, default=str)


def _export_to_excel_content(data: List[Dict[str, Any]], sheet_name: str) -> bytes:
    """Convert data to Excel format."""
    if not data:
        return b""

    df = pd.DataFrame(data)
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()


# Task monitoring functions

def get_data_task_stats():
    """Get statistics for data processing tasks."""
    try:
        inspect = app.control.inspect()

        # Get active data tasks
        active_tasks = inspect.active()
        data_tasks = []

        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    if task['name'].startswith('tasks.data_tasks'):
                        data_tasks.append({
                            'worker': worker,
                            'task_name': task['name'],
                            'task_id': task['id'],
                            'time_start': task.get('time_start')
                        })

        return {
            'active_data_tasks': len(data_tasks),
            'tasks': data_tasks,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get data task stats: {e}")
        return {'error': str(e)}