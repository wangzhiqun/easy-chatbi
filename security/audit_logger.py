"""
Audit Logger for ChatBI platform.
Comprehensive logging system for security events, user actions, and system activities.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import defaultdict, deque

from utils.logger import get_logger
from utils.config import settings

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events that can be audited."""
    LOGIN = "login"
    LOGOUT = "logout"
    QUERY_EXECUTION = "query_execution"
    DATA_ACCESS = "data_access"
    PERMISSION_CHANGE = "permission_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_CHANGE = "system_change"
    EXPORT = "export"
    IMPORT = "import"
    ERROR = "error"


class RiskLevel(Enum):
    """Risk levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: datetime
    user_id: Optional[int]
    event_type: str
    risk_level: str
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    session_id: Optional[str]
    duration_ms: Optional[int]


class AuditLogger:
    """
    Comprehensive audit logging system that tracks all security-relevant
    events and provides analysis capabilities for security monitoring.
    """

    def __init__(self):
        """Initialize audit logger with configuration."""
        self.event_queue = asyncio.Queue(maxsize=1000)
        self.event_buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds

        # Event statistics
        self.event_stats = defaultdict(int)
        self.risk_stats = defaultdict(int)
        self.user_activity = defaultdict(lambda: deque(maxlen=100))

        # Alert thresholds
        self.alert_thresholds = {
            "failed_logins": 5,
            "security_violations": 3,
            "high_risk_events": 10,
            "time_window": 300  # 5 minutes
        }

        # Start background tasks
        self._start_background_tasks()

    async def log_authentication_event(
            self,
            user_id: Optional[int],
            event_type: str,
            success: bool,
            source_ip: Optional[str] = None,
            user_agent: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log authentication-related events.

        Args:
            user_id: ID of the user (None for failed authentications)
            event_type: Type of auth event (login, logout, etc.)
            success: Whether the event was successful
            source_ip: IP address of the request
            user_agent: User agent string
            details: Additional event details

        Returns:
            Event ID of the logged event
        """
        risk_level = RiskLevel.LOW if success else RiskLevel.MEDIUM

        # Increase risk for repeated failures
        if not success and user_id:
            recent_failures = self._count_recent_events(
                user_id, "login", success=False, window_minutes=5
            )
            if recent_failures >= self.alert_thresholds["failed_logins"]:
                risk_level = RiskLevel.HIGH

        return await self.log_event(
            user_id=user_id,
            event_type=event_type,
            action="authenticate",
            success=success,
            risk_level=risk_level.value,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details or {}
        )

    async def log_data_access_event(
            self,
            user_id: int,
            resource_type: str,
            resource_id: str,
            action: str,
            success: bool,
            details: Optional[Dict[str, Any]] = None,
            duration_ms: Optional[int] = None
    ) -> str:
        """
        Log data access events.

        Args:
            user_id: ID of the user
            resource_type: Type of resource accessed (table, column, etc.)
            resource_id: Identifier of the resource
            action: Action performed (SELECT, UPDATE, etc.)
            success: Whether the action was successful
            details: Additional event details
            duration_ms: Duration of the operation

        Returns:
            Event ID of the logged event
        """
        # Determine risk level based on action and resource
        risk_level = RiskLevel.LOW

        if action in ["DELETE", "UPDATE", "DROP"]:
            risk_level = RiskLevel.HIGH
        elif resource_type == "system_table":
            risk_level = RiskLevel.MEDIUM
        elif not success:
            risk_level = RiskLevel.MEDIUM

        return await self.log_event(
            user_id=user_id,
            event_type=EventType.DATA_ACCESS.value,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            risk_level=risk_level.value,
            details=details or {},
            duration_ms=duration_ms
        )

    async def log_security_event(
            self,
            user_id: Optional[int],
            event_type: str,
            details: Dict[str, Any],
            risk_level: str = "medium",
            source_ip: Optional[str] = None,
            action: str = "security_check"
    ) -> str:
        """
        Log security-related events.

        Args:
            user_id: ID of the user (if applicable)
            event_type: Type of security event
            details: Detailed information about the event
            risk_level: Risk level of the event
            source_ip: IP address if available
            action: Action that triggered the event

        Returns:
            Event ID of the logged event
        """
        # Check for security violation patterns
        if user_id:
            recent_violations = self._count_recent_events(
                user_id, "security_violation", window_minutes=5
            )
            if recent_violations >= self.alert_thresholds["security_violations"]:
                await self._trigger_security_alert(user_id, "multiple_violations", details)

        return await self.log_event(
            user_id=user_id,
            event_type=event_type,
            action=action,
            success=False,  # Security events are typically failures
            risk_level=risk_level,
            source_ip=source_ip,
            details=details
        )

    async def log_query_execution(
            self,
            user_id: int,
            sql_query: str,
            success: bool,
            execution_time_ms: int,
            result_rows: int = 0,
            error_message: Optional[str] = None,
            session_id: Optional[str] = None
    ) -> str:
        """
        Log SQL query execution events.

        Args:
            user_id: ID of the user executing the query
            sql_query: SQL query that was executed
            success: Whether execution was successful
            execution_time_ms: Execution duration in milliseconds
            result_rows: Number of rows returned
            error_message: Error message if execution failed
            session_id: Session identifier

        Returns:
            Event ID of the logged event
        """
        # Hash the query for security (don't store full SQL)
        import hashlib
        query_hash = hashlib.sha256(sql_query.encode()).hexdigest()[:16]

        # Determine risk level
        risk_level = RiskLevel.LOW
        if not success:
            risk_level = RiskLevel.MEDIUM
        elif execution_time_ms > 30000:  # > 30 seconds
            risk_level = RiskLevel.MEDIUM
        elif result_rows > 50000:  # Large result set
            risk_level = RiskLevel.LOW

        details = {
            "query_hash": query_hash,
            "execution_time_ms": execution_time_ms,
            "result_rows": result_rows,
            "query_length": len(sql_query)
        }

        if error_message:
            details["error_message"] = error_message

        return await self.log_event(
            user_id=user_id,
            event_type=EventType.QUERY_EXECUTION.value,
            action="execute_sql",
            success=success,
            risk_level=risk_level.value,
            details=details,
            duration_ms=execution_time_ms,
            session_id=session_id
        )

    async def log_event(
            self,
            user_id: Optional[int],
            event_type: str,
            action: str,
            success: bool,
            risk_level: str = "low",
            resource_type: Optional[str] = None,
            resource_id: Optional[str] = None,
            source_ip: Optional[str] = None,
            user_agent: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            duration_ms: Optional[int] = None,
            session_id: Optional[str] = None,
            error_message: Optional[str] = None
    ) -> str:
        """
        Log a generic audit event.

        Args:
            user_id: ID of the user
            event_type: Type of event
            action: Action performed
            success: Whether the action was successful
            risk_level: Risk level of the event
            resource_type: Type of resource involved
            resource_id: ID of the resource
            source_ip: Source IP address
            user_agent: User agent string
            details: Additional event details
            duration_ms: Duration of the operation
            session_id: Session identifier
            error_message: Error message if applicable

        Returns:
            Event ID of the logged event
        """
        try:
            # Generate unique event ID
            event_id = self._generate_event_id()

            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                event_type=event_type,
                risk_level=risk_level,
                source_ip=source_ip,
                user_agent=user_agent,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                details=details or {},
                success=success,
                error_message=error_message,
                session_id=session_id,
                duration_ms=duration_ms
            )

            # Queue for processing
            await self.event_queue.put(event)

            # Update statistics
            self._update_statistics(event)

            # Check for immediate alerts
            await self._check_immediate_alerts(event)

            logger.debug(f"Audit event logged: {event_id} - {event_type}")
            return event_id

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return ""

    async def get_user_activity(
            self,
            user_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            event_types: Optional[List[str]] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit events for a specific user.

        Args:
            user_id: ID of the user
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            limit: Maximum number of events to return

        Returns:
            List of audit events
        """
        try:
            # In a real implementation, this would query the audit database
            # For now, return recent events from memory
            user_events = list(self.user_activity[user_id])

            # Apply filters
            if start_time:
                user_events = [e for e in user_events if e.timestamp >= start_time]

            if end_time:
                user_events = [e for e in user_events if e.timestamp <= end_time]

            if event_types:
                user_events = [e for e in user_events if e.event_type in event_types]

            # Sort by timestamp (newest first) and limit
            user_events.sort(key=lambda x: x.timestamp, reverse=True)
            user_events = user_events[:limit]

            # Convert to dictionaries
            return [asdict(event) for event in user_events]

        except Exception as e:
            logger.error(f"Failed to get user activity for user {user_id}: {e}")
            return []

    async def get_security_summary(
            self,
            time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get security summary for the specified time window.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Security summary statistics
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

            summary = {
                "time_window_hours": time_window_hours,
                "total_events": sum(self.event_stats.values()),
                "events_by_type": dict(self.event_stats),
                "events_by_risk": dict(self.risk_stats),
                "active_users": len(self.user_activity),
                "alerts_triggered": 0,  # Would track from alert system
                "top_risk_events": [],
                "failed_authentication_count": self.event_stats.get("failed_login", 0),
                "security_violations": self.event_stats.get("security_violation", 0)
            }

            # Calculate risk score
            risk_score = self._calculate_security_risk_score()
            summary["security_risk_score"] = risk_score

            return summary

        except Exception as e:
            logger.error(f"Failed to generate security summary: {e}")
            return {"error": str(e)}

    async def search_events(
            self,
            query: str,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            risk_levels: Optional[List[str]] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search audit events with flexible criteria.

        Args:
            query: Search query string
            start_time: Start of time range
            end_time: End of time range
            risk_levels: Filter by risk levels
            limit: Maximum number of results

        Returns:
            List of matching audit events
        """
        try:
            # In a real implementation, this would use a search engine or database query
            # For now, return a placeholder
            return [{
                "message": "Event search functionality would be implemented here",
                "query": query,
                "time_range": f"{start_time} to {end_time}",
                "filters": {"risk_levels": risk_levels, "limit": limit}
            }]

        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            return []

    # Private helper methods

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _update_statistics(self, event: AuditEvent):
        """Update event statistics."""
        self.event_stats[event.event_type] += 1
        self.risk_stats[event.risk_level] += 1

        if event.user_id:
            self.user_activity[event.user_id].append(event)

    def _count_recent_events(
            self,
            user_id: int,
            event_type: str,
            success: Optional[bool] = None,
            window_minutes: int = 5
    ) -> int:
        """Count recent events for a user."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        user_events = self.user_activity[user_id]

        count = 0
        for event in user_events:
            if (event.timestamp >= cutoff_time and
                    event.event_type == event_type and
                    (success is None or event.success == success)):
                count += 1

        return count

    async def _check_immediate_alerts(self, event: AuditEvent):
        """Check if event should trigger immediate alerts."""
        try:
            # Critical risk events
            if event.risk_level == RiskLevel.CRITICAL.value:
                await self._trigger_security_alert(
                    event.user_id,
                    "critical_risk_event",
                    {"event_id": event.event_id, "event_type": event.event_type}
                )

            # Multiple failed logins
            if (event.event_type == "login" and not event.success and event.user_id):
                recent_failures = self._count_recent_events(
                    event.user_id, "login", success=False, window_minutes=5
                )
                if recent_failures >= self.alert_thresholds["failed_logins"]:
                    await self._trigger_security_alert(
                        event.user_id,
                        "multiple_failed_logins",
                        {"failure_count": recent_failures}
                    )

        except Exception as e:
            logger.error(f"Failed to check immediate alerts: {e}")

    async def _trigger_security_alert(
            self,
            user_id: Optional[int],
            alert_type: str,
            details: Dict[str, Any]
    ):
        """Trigger a security alert."""
        try:
            alert_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "alert_type": alert_type,
                "details": details,
                "severity": "high"
            }

            # Log the alert
            logger.warning(f"Security alert triggered: {alert_type} for user {user_id}")

            # In a real implementation, this would:
            # - Send notifications to security team
            # - Update security dashboard
            # - Potentially block user account
            # - Integrate with SIEM systems

        except Exception as e:
            logger.error(f"Failed to trigger security alert: {e}")

    def _calculate_security_risk_score(self) -> int:
        """Calculate overall security risk score (0-100)."""
        try:
            base_score = 20  # Base security level

            # Add points for different risk factors
            critical_events = self.risk_stats.get(RiskLevel.CRITICAL.value, 0)
            high_events = self.risk_stats.get(RiskLevel.HIGH.value, 0)
            medium_events = self.risk_stats.get(RiskLevel.MEDIUM.value, 0)

            risk_score = (
                    base_score +
                    critical_events * 20 +
                    high_events * 10 +
                    medium_events * 5
            )

            return min(100, risk_score)

        except Exception as e:
            logger.error(f"Failed to calculate risk score: {e}")
            return 50  # Default medium risk

    def _start_background_tasks(self):
        """Start background tasks for event processing."""
        try:
            # In a real implementation, this would start async tasks for:
            # - Processing event queue
            # - Flushing events to persistent storage
            # - Cleaning up old events
            # - Generating periodic reports
            logger.info("Audit logger background tasks started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")

    async def flush_events(self):
        """Flush buffered events to persistent storage."""
        try:
            # Process all queued events
            events_to_flush = []

            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                    events_to_flush.append(event)
                except asyncio.QueueEmpty:
                    break

            if events_to_flush:
                # In a real implementation, this would write to database
                logger.info(f"Flushed {len(events_to_flush)} audit events")

        except Exception as e:
            logger.error(f"Failed to flush events: {e}")

    async def cleanup_old_events(self, retention_days: int = 90):
        """Clean up old audit events."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # In a real implementation, this would delete old events from database
            logger.info(f"Cleaned up audit events older than {cutoff_date}")

        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")

    async def export_events(
            self,
            format: str = "json",
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> str:
        """
        Export audit events in specified format.

        Args:
            format: Export format (json, csv, etc.)
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Exported data as string
        """
        try:
            # In a real implementation, this would query and format events
            export_data = {
                "export_time": datetime.utcnow().isoformat(),
                "format": format,
                "time_range": {
                    "start": start_time.isoformat() if start_time else None,
                    "end": end_time.isoformat() if end_time else None
                },
                "events": []  # Would contain actual event data
            }

            if format == "json":
                return json.dumps(export_data, indent=2)
            else:
                return "Export format not supported"

        except Exception as e:
            logger.error(f"Failed to export events: {e}")
            return f"Export failed: {str(e)}"