"""
Celery application configuration for ChatBI platform.
Handles asynchronous task processing for AI analysis and data operations.
"""

from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import os

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Create Celery application
app = Celery(
    'chatbi',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'tasks.ai_tasks',
        'tasks.data_tasks'
    ]
)

# Celery configuration
app.conf.update(
    # Task routing
    task_routes={
        'tasks.ai_tasks.analyze_query': {'queue': 'ai_processing'},
        'tasks.ai_tasks.generate_chart_suggestion': {'queue': 'ai_processing'},
        'tasks.data_tasks.export_data': {'queue': 'data_processing'},
        'tasks.data_tasks.refresh_cache': {'queue': 'maintenance'},
    },

    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task time limits
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit

    # Worker settings
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,

    # Result backend settings
    result_expires=3600,  # 1 hour
    result_compression='gzip',

    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Queue settings
    task_default_queue='default',
    task_queue_max_priority=10,
    task_default_priority=5,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-expired-cache': {
            'task': 'tasks.data_tasks.cleanup_expired_cache',
            'schedule': 3600.0,  # Every hour
        },
        'update-table-statistics': {
            'task': 'tasks.data_tasks.update_table_statistics',
            'schedule': 21600.0,  # Every 6 hours
        },
        'generate-usage-reports': {
            'task': 'tasks.data_tasks.generate_usage_report',
            'schedule': 86400.0,  # Daily
        },
    },
    beat_schedule_filename='celerybeat-schedule'
)

# Queue configuration
app.conf.task_routes = {
    # AI processing queue - higher priority, dedicated workers
    'tasks.ai_tasks.*': {
        'queue': 'ai_processing',
        'routing_key': 'ai_processing',
        'priority': 8
    },

    # Data processing queue - medium priority
    'tasks.data_tasks.export_data': {
        'queue': 'data_processing',
        'routing_key': 'data_processing',
        'priority': 6
    },

    # Maintenance queue - lower priority
    'tasks.data_tasks.cleanup_*': {
        'queue': 'maintenance',
        'routing_key': 'maintenance',
        'priority': 3
    },

    # Default queue for everything else
    '*': {
        'queue': 'default',
        'routing_key': 'default',
        'priority': 5
    }
}

# Configure Redis as message broker
if settings.celery_broker_url.startswith('redis://'):
    app.conf.broker_transport_options = {
        'visibility_timeout': 3600,
        'fanout_prefix': True,
        'fanout_patterns': True
    }


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Celery worker {sender} is ready to process tasks")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    logger.info(f"Celery worker {sender} is shutting down")


# Task base class with custom functionality
class ChatBITask(app.Task):
    """Base task class with enhanced functionality."""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task executes successfully."""
        logger.info(f"Task {self.name} [{task_id}] succeeded")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {self.name} [{task_id}] failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {self.name} [{task_id}] retrying: {exc}")


# Set custom task base class
app.Task = ChatBITask


def create_celery_app():
    """Factory function to create Celery app."""
    return app


# Utility functions for task management
def get_task_status(task_id: str):
    """Get status of a specific task."""
    try:
        result = app.AsyncResult(task_id)
        return {
            'task_id': task_id,
            'status': result.status,
            'result': result.result if result.ready() else None,
            'traceback': result.traceback,
            'date_done': result.date_done
        }
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return {'task_id': task_id, 'status': 'UNKNOWN', 'error': str(e)}


def cancel_task(task_id: str):
    """Cancel a running task."""
    try:
        app.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} cancelled")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False


def get_active_tasks():
    """Get list of active tasks."""
    try:
        inspect = app.control.inspect()
        active_tasks = inspect.active()
        return active_tasks
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        return {}


def get_worker_stats():
    """Get worker statistics."""
    try:
        inspect = app.control.inspect()
        stats = inspect.stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        return {}


def get_queue_lengths():
    """Get queue lengths."""
    try:
        inspect = app.control.inspect()
        queue_lengths = {}

        # Get reserved tasks (being processed)
        reserved = inspect.reserved()
        if reserved:
            for worker, tasks in reserved.items():
                queue_lengths[f"{worker}_reserved"] = len(tasks)

        # Get scheduled tasks
        scheduled = inspect.scheduled()
        if scheduled:
            for worker, tasks in scheduled.items():
                queue_lengths[f"{worker}_scheduled"] = len(tasks)

        return queue_lengths
    except Exception as e:
        logger.error(f"Failed to get queue lengths: {e}")
        return {}


# Health check function
def health_check():
    """Perform health check on Celery workers."""
    try:
        inspect = app.control.inspect()

        # Check if workers are responding
        stats = inspect.stats()
        if not stats:
            return {'status': 'unhealthy', 'error': 'No workers responding'}

        # Check worker ping
        ping_results = inspect.ping()
        if not ping_results:
            return {'status': 'unhealthy', 'error': 'Workers not responding to ping'}

        # Get basic metrics
        active_tasks = inspect.active() or {}
        total_active = sum(len(tasks) for tasks in active_tasks.values())

        return {
            'status': 'healthy',
            'workers': len(stats),
            'active_tasks': total_active,
            'worker_stats': stats,
            'ping_results': ping_results
        }

    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        return {'status': 'unhealthy', 'error': str(e)}


# Configuration validation
def validate_celery_config():
    """Validate Celery configuration."""
    errors = []

    # Check broker URL
    if not settings.celery_broker_url:
        errors.append("CELERY_BROKER_URL not configured")

    # Check result backend
    if not settings.celery_result_backend:
        errors.append("CELERY_RESULT_BACKEND not configured")

    # Check if Redis is accessible
    try:
        import redis
        r = redis.from_url(settings.celery_broker_url)
        r.ping()
    except Exception as e:
        errors.append(f"Cannot connect to Redis broker: {e}")

    if errors:
        logger.error(f"Celery configuration errors: {errors}")
        return False, errors

    logger.info("Celery configuration validated successfully")
    return True, []


# Initialize configuration validation
if __name__ != '__main__':
    is_valid, errors = validate_celery_config()
    if not is_valid:
        logger.warning(f"Celery configuration issues detected: {errors}")