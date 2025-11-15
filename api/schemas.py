"""Pydantic schemas for request/response models"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID

# ============ Request Models ============

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    stream: bool = False
    max_iterations: Optional[int] = 10
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the latest news about AI?",
                "stream": False,
                "max_iterations": 10
            }
        }


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "AI Research Session"
            }
        }

# ============ Response Models ============

class ActionTaken(BaseModel):
    type: str  # web_search, execute_code, etc.
    input: str
    observation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationResponse(BaseModel):
    id: UUID
    query: str
    response: Optional[str] = None
    iterations: int
    status: str
    actions_taken: List[ActionTaken]
    execution_time_ms: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    conversation_count: int
    
    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    answer: str
    iterations: int
    status: str
    actions_taken: List[ActionTaken]
    execution_time_ms: int
    message: Optional[str] = "Success"


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database: str = "connected"
    version: str

