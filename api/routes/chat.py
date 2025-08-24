from typing import List

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from core import ChatService
from core.data_service import DataService
from utils import logger
from ..database import get_db
from ..schemas import (
    ChatRequest, ChatResponse,
    ConversationCreate, ConversationResponse,
    MessageResponse
)

router = APIRouter()
chat_service = ChatService()
data_service = DataService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        if request.conversation_id:
            conversation_id = request.conversation_id
        else:
            conversation_id = chat_service.create_conversation()

        response = await chat_service.process_message(
            conversation_id,
            request.message
        )

        result = data_service.clean_numpy_types(response)

        return ChatResponse(
            conversation_id=conversation_id,
            response=result['content'],
            metadata=result.get('metadata')
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
        request: ConversationCreate,
        db: Session = Depends(get_db)
):
    try:
        conversation_id = chat_service.create_conversation()
        conversation = chat_service.get_conversation(conversation_id)

        if request.title:
            conversation['title'] = request.title

        return ConversationResponse(**conversation)

    except Exception as e:
        logger.error(f"Failed to create conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    try:
        conversation = chat_service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return ConversationResponse(**conversation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(db: Session = Depends(get_db)):
    try:
        conversations = chat_service.list_conversations()
        return [ConversationResponse(**conv) for conv in conversations]

    except Exception as e:
        logger.error(f"Failed to list conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    try:
        success = chat_service.delete_conversation(conversation_id)

        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    try:
        conversation = chat_service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = conversation.get('messages', [])
        return [MessageResponse(
            id=msg['id'],
            conversation_id=conversation_id,
            role=msg['role'],
            content=msg['content'],
            metadata=msg.get('metadata'),
            created_at=msg['timestamp']
        ) for msg in messages]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
