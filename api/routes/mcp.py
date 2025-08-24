import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from core import MCPService
from utils import logger
from ..database import get_db
from ..schemas import (
    MCPToolRequest, MCPToolResponse,
    MCPResourceRequest, MCPResourceResponse,
    MCPPromptRequest, MCPPromptResponse
)

router = APIRouter()
mcp_service = MCPService()


@router.post("/tools/execute", response_model=MCPToolResponse)
async def execute_tool(request: MCPToolRequest, db: Session = Depends(get_db)):
    try:
        result = await mcp_service.execute_tool(
            request.tool_name,
            request.arguments,
            request.session_id
        )

        return MCPToolResponse(
            status='success',
            result=result
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_tools(session_id: str = None, db: Session = Depends(get_db)):
    try:
        tools = await mcp_service.list_available_tools(session_id)
        return tools

    except Exception as e:
        logger.error(f"Failed to list tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resources/read", response_model=MCPResourceResponse)
async def read_resource(request: MCPResourceRequest, db: Session = Depends(get_db)):
    try:
        content = await mcp_service.get_resource(
            request.resource_uri,
            request.session_id
        )

        return MCPResourceResponse(
            status='success',
            content=content
        )

    except Exception as e:
        logger.error(f"Resource read failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources")
async def list_resources(session_id: str = None, db: Session = Depends(get_db)):
    try:
        resources = await mcp_service.list_available_resources(session_id)
        return resources

    except Exception as e:
        logger.error(f"Failed to list resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/execute", response_model=MCPPromptResponse)
async def execute_prompt(request: MCPPromptRequest, db: Session = Depends(get_db)):
    try:
        prompt = await mcp_service.execute_prompt(
            request.prompt_name,
            request.arguments,
            request.session_id
        )

        return MCPPromptResponse(
            status='success',
            prompt=prompt
        )

    except Exception as e:
        logger.error(f"Prompt execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts")
async def list_prompts(session_id: str = None, db: Session = Depends(get_db)):
    try:
        prompts = await mcp_service.list_available_prompts(session_id)
        return prompts

    except Exception as e:
        logger.error(f"Failed to list prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/connect")
async def connect_session(server_command: List[str], db: Session = Depends(get_db)):
    try:
        session_id = await asyncio.wait_for(
            mcp_service.connect_to_server(server_command),
            timeout=30.0
        )
        return {
            'status': 'success',
            'session_id': session_id,
            'message': 'Connected to MCP server'
        }

    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def disconnect_session(session_id: str, db: Session = Depends(get_db)):
    try:
        success = await mcp_service.disconnect_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            'status': 'success',
            'message': 'Session disconnected'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str, db: Session = Depends(get_db)):
    try:
        info = mcp_service.get_session_info(session_id)

        if not info:
            raise HTTPException(status_code=404, detail="Session not found")

        return info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(db: Session = Depends(get_db)):
    try:
        sessions = mcp_service.list_sessions()
        return {
            'sessions': sessions,
            'count': len(sessions)
        }

    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
