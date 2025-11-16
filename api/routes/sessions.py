"""Session management endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete, func
from uuid import UUID

from ..schemas import CreateSessionRequest, SessionResponse
from typing import List
from ..models import Session as DBSession, Conversation
from ..dependencies import get_session, require_user

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """Create a new session for the user"""
    session = DBSession(
        user_id=current_user.id,
        title=request.title
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return SessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        conversation_count=0
    )


@router.get("/", response_model=List[SessionResponse])
async def get_all_sessions(
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """Get all sessions for the current user"""
    result = await db.execute(
        select(DBSession)
        .where(DBSession.user_id == current_user.id)
        .order_by(DBSession.created_at.desc())
    )
    sessions = result.scalars().all()
    
    # Get conversation counts for all sessions
    session_responses = []
    for session in sessions:
        count_result = await db.execute(
            select(func.count(Conversation.id))
            .where(Conversation.session_id == session.id)
        )
        conversation_count = count_result.scalar() or 0
        
        session_responses.append(
            SessionResponse(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                conversation_count=conversation_count
            )
        )
    
    return session_responses


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_details(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """Get details of a specific session"""
    result = await db.execute(
        select(DBSession)
        .where(
            and_(
                DBSession.id == session_id,
                DBSession.user_id == current_user.id
            )
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Count conversations for this session
    count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.session_id == session_id)
    )
    conversation_count = count_result.scalar() or 0
    
    return SessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        conversation_count=conversation_count
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """Delete a session and all its conversations"""
    result = await db.execute(
        select(DBSession)
        .where(
            and_(
                DBSession.id == session_id,
                DBSession.user_id == current_user.id
            )
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    await db.delete(session)
    await db.commit()

