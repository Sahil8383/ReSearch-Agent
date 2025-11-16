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
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """
    Send a query to the agent and get a response.
    
    - **query**: The question or task for the agent
    - **stream**: Enable streaming response (SSE)
    - **max_iterations**: Maximum iterations for the agent loop
    - **session_id**: Optional session ID. If not provided, a new session will be created automatically.
    
    Every conversation is automatically attached to a session for tracking purposes.
    """
    try:
        start_time = time.time()
        
        # Get or create session - every conversation requires a session
        session_id_str = str(request.session_id) if request.session_id else None
        db_session = await get_or_create_session(
            db, current_user.id, session_id_str
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
        
        # Create conversation - session_id is required (enforced by DB schema)
        conversation = Conversation(
            session_id=db_session.id,  # Every conversation must have a session_id
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
        await db.refresh(conversation)
        
        return ChatResponse(
            answer=result["answer"],
            iterations=result["iterations"],
            status=result["status"],
            actions_taken=result["actions_taken"],
            execution_time_ms=int((time.time() - start_time) * 1000),
            session_id=db_session.id,  # Return session_id so client knows which session was used
            conversation_id=conversation.id  # Return conversation_id for reference
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stream", response_class=StreamingResponse)
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_session),
    current_user = Depends(require_user)
):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Sends agent thoughts and actions in real-time.
    
    - **query**: The question or task for the agent
    - **max_iterations**: Maximum iterations for the agent loop
    - **session_id**: Optional session ID. If not provided, a new session will be created automatically.
    
    Every conversation is automatically attached to a session for tracking purposes.
    """
    async def event_generator():
        start_time = time.time()
        final_answer = None
        final_status = "pending"
        final_iterations = 0
        collected_actions = []
        db_session = None
        
        try:
            # Get or create session - every conversation requires a session
            session_id_str = str(request.session_id) if request.session_id else None
            db_session = await get_or_create_session(
                db, current_user.id, session_id_str
            )
            agent_service = AgentService()
            
            # Stream events and capture final result
            async for event in agent_service.run_agent_streaming(
                query=request.query,
                session=db_session,
                max_iterations=request.max_iterations
            ):
                # Yield the event to the client
                yield f"data: {event}\n\n"
                
                # Parse event to capture final result
                try:
                    event_data = json.loads(event)
                    event_type = event_data.get("type")
                    
                    # Capture final answer
                    if event_type == "final_answer_complete":
                        final_answer = event_data.get("answer", "")
                        final_status = event_data.get("status", "completed")
                        final_iterations = event_data.get("iterations", 0)
                    
                    # Capture actions
                    elif event_type == "action":
                        collected_actions.append({
                            "type": event_data.get("action_type", ""),
                            "input": event_data.get("input", ""),
                            "observation": None  # Will be updated when observation arrives
                        })
                    
                    # Update observation for the last action
                    elif event_type == "observation":
                        if collected_actions:
                            collected_actions[-1]["observation"] = event_data.get("observation", "")
                    
                    # Capture end event for iterations and status
                    elif event_type == "end":
                        if final_iterations == 0:
                            final_iterations = event_data.get("iterations", 0)
                        if final_status == "pending":
                            final_status = event_data.get("status", "completed")
                    
                    # Handle errors
                    elif event_type == "error":
                        if final_status == "pending":
                            final_status = "failed"
                
                except (json.JSONDecodeError, KeyError):
                    # If event is not JSON or missing keys, continue
                    pass
            
            # After streaming completes, save conversation to database
            # Ensure we have a session (should always be created above)
            if not db_session:
                session_id_str = str(request.session_id) if request.session_id else None
                db_session = await get_or_create_session(
                    db, current_user.id, session_id_str
                )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Convert actions to dict format for storage
            actions_taken_dicts = []
            for action in collected_actions:
                action_dict = {
                    "type": action.get("type", ""),
                    "input": action.get("input", ""),
                    "observation": action.get("observation"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                actions_taken_dicts.append(action_dict)
            
            # Create conversation record - session_id is required (enforced by DB schema)
            conversation = Conversation(
                session_id=db_session.id,  # Every conversation must have a session_id
                query=request.query,
                response=final_answer or "",
                iterations=final_iterations,
                status=final_status if final_status != "pending" else "failed",
                actions_taken=actions_taken_dicts,
                execution_time_ms=execution_time_ms,
                completed_at=datetime.utcnow() if final_status in ["completed", "max_iterations_reached"] else None
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            
            # Send final event with conversation info
            final_event = json.dumps({
                "type": "conversation_saved",
                "conversation_id": str(conversation.id),
                "session_id": str(db_session.id),
                "message": "Conversation saved to database"
            })
            yield f"data: {final_event}\n\n"
                
        except Exception as e:
            # Try to save conversation even on error
            try:
                if not db_session:
                    session_id_str = str(request.session_id) if request.session_id else None
                    db_session = await get_or_create_session(
                        db, current_user.id, session_id_str
                    )
                
                execution_time_ms = int((time.time() - start_time) * 1000)
                error_conversation = Conversation(
                    session_id=db_session.id,
                    query=request.query,
                    response="",
                    iterations=0,
                    status="failed",
                    actions_taken=[],
                    execution_time_ms=execution_time_ms,
                    error_message=str(e)
                )
                db.add(error_conversation)
                await db.commit()
            except Exception as save_error:
                # If we can't save, at least log it
                pass
            
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

