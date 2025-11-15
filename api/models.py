"""SQLAlchemy database models"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_username", "username"),
    )


class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    short_term_memory = Column(JSON, default=list)  # Store last 5 messages
    metadata_info = Column(JSON, default=dict)  # Custom metadata
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_session_user_id", "user_id"),
        Index("idx_session_created_at", "created_at"),
    )


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    
    # Query and Response
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    
    # Execution Details
    iterations = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, completed, failed, timeout
    error_message = Column(Text, nullable=True)
    
    # Actions Taken
    actions_taken = Column(JSON, default=list)  # [{type: "web_search", input: "...", observation: "..."}]
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Execution Time
    execution_time_ms = Column(Integer, nullable=True)
    
    # Relationship
    session = relationship("Session", back_populates="conversations")
    
    __table_args__ = (
        Index("idx_conversation_session_id", "session_id"),
        Index("idx_conversation_status", "status"),
        Index("idx_conversation_created_at", "created_at"),
    )


class AgentLog(Base):
    __tablename__ = "agent_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True)
    
    log_level = Column(String(50))  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text)
    context = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_agent_log_session_id", "session_id"),
        Index("idx_agent_log_created_at", "created_at"),
        Index("idx_agent_log_level", "log_level"),
    )

