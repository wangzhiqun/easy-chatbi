from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils import logger, get_config
from .database import init_db
from .routes import chat, data, mcp

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ChatBI API server...")
    init_db()
    yield
    logger.info("Shutting down ChatBI API server...")


app = FastAPI(
    title="ChatBI API",
    description="Data Intelligence Platform API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=f"{config.api_prefix}/chat", tags=["Chat"])
app.include_router(data.router, prefix=f"{config.api_prefix}/data", tags=["Data"])
app.include_router(mcp.router, prefix=f"{config.api_prefix}/mcp", tags=["MCP"])


@app.get("/")
async def root():
    return {
        "name": "ChatBI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ChatBI API",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower()
    )
