"""Agent service for running ReAct agent"""

from ..agent_wrapper import ReActAgentAPI
from ..schemas import ActionTaken
import asyncio
import time
import json
from datetime import datetime


class AgentService:
    def __init__(self):
        self.agent = ReActAgentAPI(stream=False)
    
    async def run_agent(self, query: str, session, max_iterations: int, db):
        """Run agent and return structured result"""
        # Set max_iterations if provided
        if max_iterations:
            self.agent.max_iterations = max_iterations
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.agent.run(query)
            )
            
            # Format actions for response
            actions_taken = [
                ActionTaken(
                    type=action["type"],
                    input=action["input"],
                    observation=action.get("observation"),
                    timestamp=datetime.utcnow()
                )
                for action in result.get("actions", [])
            ]
            
            return {
                "answer": result.get("answer", ""),
                "iterations": result.get("iterations", 0),
                "status": result.get("status", "completed"),
                "actions_taken": actions_taken
            }
        except Exception as e:
            return {
                "answer": None,
                "iterations": 0,
                "status": "failed",
                "error": str(e),
                "actions_taken": []
            }
    
    async def _stream_llm_response(self, agent):
        """Stream LLM response token by token"""
        loop = asyncio.get_event_loop()
        token_queue = asyncio.Queue()
        
        def _run_stream():
            """Run the synchronous stream and put tokens in the async queue"""
            # Enable streaming on the agent temporarily
            original_stream = agent.stream
            agent.stream = True
            
            try:
                stream = agent.client.messages.create(
                    model=agent.model,
                    system=agent.system_prompt,
                    messages=agent.get_messages_for_api(),
                    temperature=0.3,
                    max_tokens=500,
                    stream=True
                )
                
                # The stream is synchronous, so we iterate it here
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta.text
                        # Put token in queue (this will be awaited by the async consumer)
                        asyncio.run_coroutine_threadsafe(token_queue.put(delta), loop)
            except Exception as e:
                # Put error in queue
                asyncio.run_coroutine_threadsafe(token_queue.put(f"__ERROR__:{str(e)}"), loop)
            finally:
                # Restore original stream setting
                agent.stream = original_stream
                # Signal completion
                asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)
        
        # Start streaming in executor
        loop.run_in_executor(None, _run_stream)
        
        # Yield tokens as they arrive from the queue
        while True:
            token = await token_queue.get()
            if token is None:
                break
            if isinstance(token, str) and token.startswith("__ERROR__:"):
                raise Exception(token.replace("__ERROR__:", ""))
            yield token
    
    async def run_agent_streaming(self, query: str, session):
        """Stream agent execution with real-time updates"""
        # Create a fresh agent instance for streaming
        streaming_agent = ReActAgentAPI(stream=False)
        streaming_agent.max_iterations = 10
        
        # Reset and add user query
        streaming_agent.reset()
        streaming_agent.add_message("user", query)
        
        yield json.dumps({"type": "start", "message": "Starting agent...", "query": query})
        
        iteration = 0
        final_answer = None
        status = "completed"
        loop = asyncio.get_event_loop()
        
        while iteration < streaming_agent.max_iterations:
            iteration += 1
            
            yield json.dumps({
                "type": "iteration_start",
                "iteration": iteration,
                "message": f"Starting iteration {iteration}"
            })
            
            # Get response from LLM with token-by-token streaming
            yield json.dumps({"type": "thinking", "message": "Agent is thinking..."})
            yield json.dumps({"type": "thought_start", "iteration": iteration})
            
            try:
                # Stream the LLM response token by token
                full_response = ""
                async for token in self._stream_llm_response(streaming_agent):
                    full_response += token
                    # Yield each token as it arrives
                    yield json.dumps({
                        "type": "thought_token",
                        "token": token,
                        "iteration": iteration
                    })
                
                # Signal that thought is complete
                yield json.dumps({
                    "type": "thought_complete",
                    "iteration": iteration
                })
                
                streaming_agent.add_message("assistant", full_response)
                
                # Check for actions
                action = streaming_agent.parse_action(full_response)
                
                if action:
                    action_type, action_input = action
                    
                    yield json.dumps({
                        "type": "action",
                        "action_type": action_type,
                        "input": action_input,
                        "iteration": iteration
                    })
                    
                    # Execute the action
                    # Capture action variables explicitly to avoid closure issues
                    captured_action_type = action_type
                    captured_action_input = action_input
                    
                    def _execute_action():
                        return streaming_agent.execute_action(captured_action_type, captured_action_input)
                    
                    observation = await loop.run_in_executor(None, _execute_action)
                    
                    # Store action
                    streaming_agent.collected_actions.append({
                        "type": action_type,
                        "input": action_input,
                        "observation": observation
                    })
                    
                    yield json.dumps({
                        "type": "observation",
                        "action_type": action_type,
                        "observation": observation[:1000] if len(observation) > 1000 else observation,  # Truncate long observations
                        "iteration": iteration
                    })
                    
                    # Add observation to messages
                    observation_text = f"Observation from {action_type}: {observation}"
                    streaming_agent.add_message("user", observation_text)
                    
                    continue
                
                # Check for final answer
                if streaming_agent.is_final_answer(full_response):
                    final_answer = streaming_agent.extract_final_answer(full_response)
                    
                    # The final answer was already streamed as tokens above
                    # Just signal completion
                    yield json.dumps({
                        "type": "final_answer_complete",
                        "answer": final_answer,
                        "iterations": iteration,
                        "status": "completed"
                    })
                    
                    status = "completed"
                    break
                
                # No action and no final answer
                if "PAUSE" not in full_response:
                    streaming_agent.add_message("user", "Please continue with a Thought and Action, or provide a Final Answer.")
            
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "message": f"Error in iteration {iteration}: {str(e)}"
                })
                status = "failed"
                break
        
        # Max iterations reached
        if iteration >= streaming_agent.max_iterations and not final_answer:
            yield json.dumps({
                "type": "max_iterations",
                "message": "Maximum iterations reached. Generating final answer..."
            })
            
            streaming_agent.add_message("user", "Please provide a Final Answer based on the information gathered so far.")
            
            try:
                # Stream the final answer token by token
                yield json.dumps({"type": "final_answer_start"})
                
                final_response = ""
                async for token in self._stream_llm_response(streaming_agent):
                    final_response += token
                    # Yield each token as it arrives
                    yield json.dumps({
                        "type": "final_answer_token",
                        "token": token
                    })
                
                final_answer = streaming_agent.extract_final_answer(final_response)
                status = "max_iterations_reached"
                
                yield json.dumps({
                    "type": "final_answer_complete",
                    "answer": final_answer,
                    "iterations": iteration,
                    "status": status
                })
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "message": f"Error generating final answer: {str(e)}"
                })
        
        # End event - don't include full answer since it was already sent in final_answer_complete
        yield json.dumps({
            "type": "end",
            "iterations": iteration,
            "status": status,
            "actions_count": len(streaming_agent.collected_actions)
        })
    
    def _format_actions(self, messages):
        """Extract and format actions from messages"""
        actions = []
        for msg in messages:
            # Parse actions from agent messages
            if "Action:" in msg.content:
                actions.append(ActionTaken(
                    type="parsed_action",
                    input=msg.content
                ))
        return actions

