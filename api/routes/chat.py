"""Chat endpoints for the agent API"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import time
import json
from datetime import datetime

from ..schemas import ChatRequest, ChatResponse, ErrorResponse, ActionTaken
from ..dependencies import get_session, get_or_create_session, require_user
from ..services.agent_service import AgentService
from ..models import Conversation, Session as DBSession

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session_id: str = None,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """
    Send a query to the agent and get a response.
    
    - **query**: The question or task for the agent
    - **stream**: Enable streaming response (SSE)
    - **max_iterations**: Maximum iterations for the agent loop
    """
    try:
        start_time = time.time()
        
        # Get or create session
        db_session = await get_or_create_session(
            db, current_user.id, session_id
        )
        
        # Initialize agent service
        agent_service = AgentService()
        
        # Run agent
        result = await agent_service.run_agent(
            query=request.query,
            session=db_session,
            max_iterations=request.max_iterations,
            db=db
        )
        
        # Store conversation - convert ActionTaken models to dicts and serialize datetimes
        actions_taken_dicts = []
        for action in result["actions_taken"]:
            action_dict = action.model_dump() if hasattr(action, 'model_dump') else dict(action)
            # Convert datetime to ISO string for JSON storage
            if 'timestamp' in action_dict and isinstance(action_dict['timestamp'], datetime):
                action_dict['timestamp'] = action_dict['timestamp'].isoformat()
            actions_taken_dicts.append(action_dict)
        
        conversation = Conversation(
            session_id=db_session.id,
            query=request.query,
            response=result["answer"],
            iterations=result["iterations"],
            status=result["status"],
            actions_taken=actions_taken_dicts,
            execution_time_ms=int((time.time() - start_time) * 1000),
            completed_at=datetime.utcnow()
        )
        db.add(conversation)
        await db.commit()
        
        return ChatResponse(
            answer=result["answer"],
            iterations=result["iterations"],
            status=result["status"],
            actions_taken=result["actions_taken"],
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stream", response_class=StreamingResponse)
async def chat_stream(
    request: ChatRequest,
    session_id: str = None,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Sends agent thoughts and actions in real-time.
    
    - **query**: The question or task for the agent
    - **max_iterations**: Maximum iterations for the agent loop
    """
    async def event_generator():
        try:
            db_session = await get_or_create_session(
                db, current_user.id, session_id
            )
            agent_service = AgentService()
            
            async for event in agent_service.run_agent_streaming(
                query=request.query,
                session=db_session,
                max_iterations=request.max_iterations
            ):
                yield f"data: {event}\n\n"
                
        except Exception as e:
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

