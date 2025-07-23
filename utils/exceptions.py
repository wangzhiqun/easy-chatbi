"""
Custom exceptions for ChatBI platform.
Defines specific exception types for different error scenarios.
"""

from typing import Optional, Any


class ChatBIException(Exception):
    """Base exception class for ChatBI platform."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class DatabaseException(ChatBIException):
    """Exception raised for database-related errors."""
    pass


class AuthenticationException(ChatBIException):
    """Exception raised for authentication failures."""
    pass


class AuthorizationException(ChatBIException):
    """Exception raised for authorization failures."""
    pass


class SQLSecurityException(ChatBIException):
    """Exception raised for SQL security violations."""
    pass


class LLMException(ChatBIException):
    """Exception raised for LLM-related errors."""
    pass


class DataProcessingException(ChatBIException):
    """Exception raised for data processing errors."""
    pass


class ConfigurationException(ChatBIException):
    """Exception raised for configuration errors."""
    pass


class ValidationException(ChatBIException):
    """Exception raised for data validation errors."""
    pass


class ServiceUnavailableException(ChatBIException):
    """Exception raised when external services are unavailable."""
    pass


class RateLimitException(ChatBIException):
    """Exception raised when rate limits are exceeded."""
    pass


# Error code constants
class ErrorCodes:
    """Standard error codes for the platform."""

    # Database errors
    DB_CONNECTION_ERROR = "DB_001"
    DB_QUERY_ERROR = "DB_002"
    DB_TRANSACTION_ERROR = "DB_003"

    # Authentication errors
    AUTH_INVALID_CREDENTIALS = "AUTH_001"
    AUTH_TOKEN_EXPIRED = "AUTH_002"
    AUTH_TOKEN_INVALID = "AUTH_003"

    # Authorization errors
    AUTHZ_INSUFFICIENT_PERMISSIONS = "AUTHZ_001"
    AUTHZ_RESOURCE_FORBIDDEN = "AUTHZ_002"

    # SQL security errors
    SQL_INJECTION_DETECTED = "SQL_001"
    SQL_UNAUTHORIZED_OPERATION = "SQL_002"
    SQL_DANGEROUS_QUERY = "SQL_003"

    # LLM errors
    LLM_API_ERROR = "LLM_001"
    LLM_QUOTA_EXCEEDED = "LLM_002"
    LLM_INVALID_RESPONSE = "LLM_003"

    # Data processing errors
    DATA_VALIDATION_ERROR = "DATA_001"
    DATA_PARSING_ERROR = "DATA_002"
    DATA_TRANSFORMATION_ERROR = "DATA_003"

    # Configuration errors
    CONFIG_MISSING_SETTING = "CONFIG_001"
    CONFIG_INVALID_VALUE = "CONFIG_002"

    # Service errors
    SERVICE_UNAVAILABLE = "SVC_001"
    SERVICE_TIMEOUT = "SVC_002"
    SERVICE_RATE_LIMITED = "SVC_003"