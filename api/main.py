"""
FastAPI main application for ChatBI platform.
Configures the API server with routes, middleware, and error handling.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import time
import uvicorn

from utils.config import settings
from utils.logger import get_logger
from utils.exceptions import ChatBIException
from .database import health_check, test_database_connection, test_redis_connection
from .routes import auth, chat, data
from . import models

logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="ChatBI - Intelligent Data Analytics Platform",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.api_host]
    )


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(ChatBIException)
async def chatbi_exception_handler(request: Request, exc: ChatBIException):
    """Handle custom ChatBI exceptions."""
    logger.error(f"ChatBI exception: {exc.message}", extra={"error_code": exc.error_code})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom format."""
    logger.error(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Test database connection
    if not test_database_connection():
        logger.error("Failed to connect to database")
        raise Exception("Database connection failed")

    # Test Redis connection
    if not test_redis_connection():
        logger.warning("Failed to connect to Redis")

    logger.info("Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Application shutting down")


# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs" if settings.debug else "disabled"
    }


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        health_status = await health_check()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": health_status,
            "version": settings.app_version
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e),
                "version": settings.app_version
            }
        )


# Metrics endpoint (for monitoring)
@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint."""
    return {
        "uptime": "N/A",  # Would implement proper uptime tracking
        "requests_total": "N/A",  # Would implement request counting
        "response_time_avg": "N/A",  # Would implement response time tracking
        "active_sessions": "N/A"  # Would implement session tracking
    }


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )