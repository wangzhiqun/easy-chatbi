from typing import Optional, Any, Dict


class ChatBIException(Exception):
    def __init__(
            self,
            message: str,
            code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code or "CHATBI_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details
        }


class DatabaseError(ChatBIException):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class AIError(ChatBIException):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AI_ERROR", details)


class MCPError(ChatBIException):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MCP_ERROR", details)


class ValidationError(ChatBIException):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class AuthenticationError(ChatBIException):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTH_ERROR", details)


class ConfigurationError(ChatBIException):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIG_ERROR", details)


class MCPAgentError(ChatBIException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MCP_AGENT_ERROR", details)
