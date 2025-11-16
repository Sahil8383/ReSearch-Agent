"""Dependency injection for FastAPI routes"""

from fastapi import HTTPException, status, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from .database import get_session as get_db_session
from .models import User, Session as DBSession

security = HTTPBearer()


async def get_session(db: AsyncSession = Depends(get_db_session)) -> AsyncSession:
    """Get database session"""
    return db


async def require_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_session)
):
    """Validate and get current user"""
    # This is simplified - implement proper JWT validation
    try:
        user_id = credentials.credentials  # Would be extracted from JWT
        result = await db.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


async def get_or_create_session(
    db: AsyncSession,
    user_id: UUID,
    session_id: str = None
):
    """Get existing session or create new one"""
    if session_id:
        result = await db.execute(
            select(DBSession).where(
                (DBSession.id == UUID(session_id)) &
                (DBSession.user_id == user_id)
            )
        )
        session = result.scalar_one_or_none()
        if session:
            # Refresh to get latest data including short_term_memory
            await db.refresh(session)
            return session
    
    # Create new session
    session = DBSession(user_id=user_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session

