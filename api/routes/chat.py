"""
Chat routes for ChatBI platform.
Handles conversation management and AI-powered query processing.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..database import get_db
from ..models import User as UserModel, ChatSession as ChatSessionModel, ChatMessage as ChatMessageModel
from ..schemas import (
    ChatSession, ChatSessionCreate, ChatSessionUpdate, ChatSessionWithMessages,
    ChatMessage, ChatMessageCreate, QueryRequest, QueryResponse
)
from .auth import get_current_active_user
from services.chat_service import ChatService
from utils.logger import get_logger
from utils.exceptions import ChatBIException, ErrorCodes

logger = get_logger(__name__)

router = APIRouter()


# Dependency injection
def get_chat_service() -> ChatService:
    """Get chat service instance."""
    return ChatService()


@router.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100)
):
    """Get user's chat sessions."""
    sessions = db.query(ChatSessionModel).filter(
        ChatSessionModel.user_id == current_user.id
    ).order_by(desc(ChatSessionModel.updated_at)).offset(skip).limit(limit).all()

    # Add message count to each session
    result = []
    for session in sessions:
        session_dict = session.__dict__.copy()
        session_dict['message_count'] = db.query(ChatMessageModel).filter(
            ChatMessageModel.session_id == session.id
        ).count()
        result.append(ChatSession(**session_dict))

    return result


@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(
        session_data: ChatSessionCreate,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Create a new chat session."""
    try:
        session = ChatSessionModel(
            user_id=current_user.id,
            session_name=session_data.session_name
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        logger.info(f"New chat session created: {session.id} for user: {current_user.username}")
        return ChatSession.from_orm(session)

    except Exception as e:
        logger.error(f"Failed to create chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session"
        )


@router.get("/sessions/{session_id}", response_model=ChatSessionWithMessages)
async def get_chat_session(
        session_id: int,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get specific chat session with messages."""
    session = db.query(ChatSessionModel).filter(
        ChatSessionModel.id == session_id,
        ChatSessionModel.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Get messages for this session
    messages = db.query(ChatMessageModel).filter(
        ChatMessageModel.session_id == session_id
    ).order_by(ChatMessageModel.created_at).all()

    session_data = ChatSessionWithMessages.from_orm(session)
    session_data.messages = [ChatMessage.from_orm(msg) for msg in messages]

    return session_data


@router.put("/sessions/{session_id}", response_model=ChatSession)
async def update_chat_session(
        session_id: int,
        session_data: ChatSessionUpdate,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Update chat session."""
    session = db.query(ChatSessionModel).filter(
        ChatSessionModel.id == session_id,
        ChatSessionModel.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Update session fields
    update_data = session_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)

    db.commit()
    db.refresh(session)

    return ChatSession.from_orm(session)


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
        session_id: int,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Delete chat session and all its messages."""
    session = db.query(ChatSessionModel).filter(
        ChatSessionModel.id == session_id,
        ChatSessionModel.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Delete all messages in the session
    db.query(ChatMessageModel).filter(
        ChatMessageModel.session_id == session_id
    ).delete()

    # Delete the session
    db.delete(session)
    db.commit()

    logger.info(f"Chat session deleted: {session_id} by user: {current_user.username}")
    return {"message": "Chat session deleted successfully"}


@router.post("/sessions/{session_id}/messages", response_model=ChatMessage)
async def add_message_to_session(
        session_id: int,
        message_data: ChatMessageCreate,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Add a message to a chat session."""
    # Verify session belongs to user
    session = db.query(ChatSessionModel).filter(
        ChatSessionModel.id == session_id,
        ChatSessionModel.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Create message
    message = ChatMessageModel(
        session_id=session_id,
        message_type=message_data.message_type,
        content=message_data.content,
        metadata=message_data.metadata
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    # Update session timestamp
    session.updated_at = message.created_at
    db.commit()

    return ChatMessage.from_orm(message)


@router.post("/query", response_model=QueryResponse)
async def process_query(
        query_request: QueryRequest,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        chat_service: ChatService = Depends(get_chat_service)
):
    """Process natural language query and return SQL with results."""
    try:
        # Validate session if provided
        session = None
        if query_request.session_id:
            session = db.query(ChatSessionModel).filter(
                ChatSessionModel.id == query_request.session_id,
                ChatSessionModel.user_id == current_user.id
            ).first()

            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )

        # Process the query using chat service
        response = await chat_service.process_query(
            user_id=current_user.id,
            question=query_request.question,
            session_id=query_request.session_id,
            include_chart_suggestion=query_request.include_chart_suggestion,
            db=db
        )

        logger.info(f"Query processed for user {current_user.username}: {query_request.question[:50]}...")
        return response

    except ChatBIException as e:
        logger.error(f"Query processing failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )


@router.post("/chat", response_model=QueryResponse)
async def chat_with_ai(
        query_request: QueryRequest,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        chat_service: ChatService = Depends(get_chat_service)
):
    """Chat with AI and get contextual responses."""
    try:
        # If no session provided, create one
        if not query_request.session_id:
            session = ChatSessionModel(
                user_id=current_user.id,
                session_name=f"Chat {query_request.question[:30]}..."
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            query_request.session_id = session.id

        # Add user message to session
        user_message = ChatMessageModel(
            session_id=query_request.session_id,
            message_type="user",
            content=query_request.question
        )
        db.add(user_message)
        db.commit()

        # Process query
        response = await chat_service.process_query(
            user_id=current_user.id,
            question=query_request.question,
            session_id=query_request.session_id,
            include_chart_suggestion=query_request.include_chart_suggestion,
            db=db
        )

        # Add AI response to session
        ai_message = ChatMessageModel(
            session_id=query_request.session_id,
            message_type="assistant",
            content=response.generated_sql if response.generated_sql else "I couldn't generate a query for that request.",
            metadata={
                "execution_status": response.execution_status,
                "execution_time_ms": response.execution_time_ms,
                "result_rows": len(response.result_data) if response.result_data else 0
            }
        )
        db.add(ai_message)
        db.commit()

        return response

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
        session_id: int,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100)
):
    """Get messages from a specific chat session."""
    # Verify session belongs to user
    session = db.query(ChatSessionModel).filter(
        ChatSessionModel.id == session_id,
        ChatSessionModel.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    messages = db.query(ChatMessageModel).filter(
        ChatMessageModel.session_id == session_id
    ).order_by(ChatMessageModel.created_at).offset(skip).limit(limit).all()

    return [ChatMessage.from_orm(msg) for msg in messages]