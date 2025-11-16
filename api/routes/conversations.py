"""Conversation management endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import UUID
from typing import List, Optional
from datetime import datetime

from ..schemas import ConversationResponse, ActionTaken
from ..models import Conversation, Session as DBSession
from ..dependencies import get_session, require_user

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def convert_actions_taken(actions_data: List[dict]) -> List[ActionTaken]:
    """Convert JSON actions_taken data to ActionTaken objects"""
    actions = []
    for action_dict in actions_data:
        # Handle timestamp conversion if it's a string
        if 'timestamp' in action_dict and isinstance(action_dict['timestamp'], str):
            action_dict['timestamp'] = datetime.fromisoformat(action_dict['timestamp'])
        actions.append(ActionTaken(**action_dict))
    return actions


@router.get("/", response_model=List[ConversationResponse])
async def get_conversations(
    session_id: UUID = Query(..., description="Session ID to get conversations for"),
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """Get all conversations for a specific session"""
    # First verify the session exists and belongs to the user
    session_result = await db.execute(
        select(DBSession)
        .where(
            and_(
                DBSession.id == session_id,
                DBSession.user_id == current_user.id
            )
        )
    )
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get all conversations for this session
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
    )
    conversations = result.scalars().all()
    
    # Convert to response models
    conversation_responses = []
    for conv in conversations:
        # Convert actions_taken from JSON to ActionTaken objects
        actions_taken = convert_actions_taken(conv.actions_taken or [])
        
        conversation_responses.append(
            ConversationResponse(
                id=conv.id,
                session_id=conv.session_id,  # Every conversation has a session_id
                query=conv.query,
                response=conv.response,
                iterations=conv.iterations,
                status=conv.status,
                actions_taken=actions_taken,
                execution_time_ms=conv.execution_time_ms,
                created_at=conv.created_at
            )
        )
    
    return conversation_responses


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """Get a specific conversation by ID"""
    # Get the conversation with its session
    result = await db.execute(
        select(Conversation)
        .join(DBSession)
        .where(
            and_(
                Conversation.id == conversation_id,
                DBSession.user_id == current_user.id
            )
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Convert actions_taken from JSON to ActionTaken objects
    actions_taken = convert_actions_taken(conversation.actions_taken or [])
    
    return ConversationResponse(
        id=conversation.id,
        session_id=conversation.session_id,  # Every conversation has a session_id
        query=conversation.query,
        response=conversation.response,
        iterations=conversation.iterations,
        status=conversation.status,
        actions_taken=actions_taken,
        execution_time_ms=conversation.execution_time_ms,
        created_at=conversation.created_at
    )

