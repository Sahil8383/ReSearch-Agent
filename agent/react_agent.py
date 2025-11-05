"""ReAct Agent implementation"""

import os
import re
from typing import List, Dict, Optional
from anthropic import Anthropic
from dotenv import load_dotenv

from agent.message import Message
from agent.tools import WebSearchTool, CodeExecutionTool

# Load environment variables
load_dotenv()


class ReActAgent:
    """ReAct Agent that thinks, acts, and observes in a loop"""
    
    def __init__(self, model: str = "claude-3-5-haiku-20241022", max_iterations: int = 10, stream: bool = True):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.stream = stream
        self.messages: List[Message] = []
        self.web_search = WebSearchTool()
        self.code_executor = CodeExecutionTool()
        
        # System prompt that defines the agent's behavior
        self.system_prompt = """You are a ReAct (Reasoning + Acting) agent that follows a strict step-by-step process.

CRITICAL: You can ONLY output ONE of the following in each response:
- Thought + Action (then PAUSE)
- OR Final Answer (only after receiving observations)

You run in a loop:
1. Think about what you need
2. Take ONE action
3. Wait for the observation
4. React to the observation
5. Repeat until you have enough information

Available Actions:
- web_search: <query> - Search the web for information
- execute_code: <python code> - Execute Python code and get the output

Format for Action:
Thought: <your reasoning>
Action: web_search: <your search query>
PAUSE

OR for code execution:
Thought: <your reasoning>
Action: execute_code: <python code>
PAUSE

Note: For execute_code, the code can be multi-line. Everything after "execute_code:" until "PAUSE" will be treated as Python code.

Format for Final Answer (ONLY after receiving observations):
Thought: <your reasoning based on observations>
Final Answer: <your complete answer>

ABSOLUTE RULES - DO NOT VIOLATE:
1. ONE action per response - NEVER multiple actions
2. NEVER include Final Answer in the same response as an Action
3. NEVER generate your own Observation - wait for the tool
4. After PAUSE, you will receive an Observation - then you can think again
5. Only provide Final Answer after you have received and reviewed actual Observations
6. Stop immediately after Action + PAUSE - do not continue planning

BAD EXAMPLE (DO NOT DO THIS):
Thought: I need to search
Action: web_search: query1
PAUSE
Thought: I also need to search
Action: web_search: query2
PAUSE
Final Answer: answer

GOOD EXAMPLE:
Response 1:
Thought: I need to search for information about X
Action: web_search: information about X
PAUSE

[After receiving Observation]

Response 2:
Thought: Based on the observation, I need more specific info about Y
Action: web_search: specific info about Y
PAUSE

[After receiving Observation]

Response 3:
Thought: I now have enough information from the observations
Final Answer: [your answer based on the observations]
"""
    
    def reset(self):
        """Reset the agent's memory"""
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append(Message(role, content))
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Convert messages to format required by Anthropic API"""
        return [msg.to_dict() for msg in self.messages]
    
    def call_llm(self) -> str:
        """Call the LLM and get a response (streaming or non-streaming)"""
        try:
            if self.stream:
                # Streaming mode
                stream = self.client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=self.get_messages_for_api(),
                    temperature=0.3,
                    max_tokens=500,
                    stream=True
                )
                
                full_response = ""
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta.text
                        full_response += delta
                        print(delta, end="", flush=True)
                
                return full_response
            else:
                # Non-streaming mode
                response = self.client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=self.get_messages_for_api(),
                    temperature=0.3,  # Lower temperature for more focused, step-by-step responses
                    max_tokens=500  # Limit tokens to prevent planning ahead
                )
                return response.content[0].text
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def parse_action(self, text: str) -> Optional[tuple]:
        """Parse action from LLM response - only gets FIRST action"""
        # Look for Action: web_search: <query>
        web_search_pattern = r"Action:\s*web_search:\s*(.+?)(?:\n|PAUSE|$)"
        match = re.search(web_search_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            query = match.group(1).strip()
            # Clean up query - remove any trailing thoughts or actions
            query = query.split('\n')[0].split('PAUSE')[0].strip()
            return ("web_search", query)
        
        # Look for Action: execute_code: <code>
        # Code can be multi-line, so we capture everything until PAUSE or end
        # The code can start on the same line or on a new line after "execute_code:"
        code_pattern = r"Action:\s*execute_code:\s*(.*?)(?=\n\s*PAUSE\s*|PAUSE\s*$|$)"
        match = re.search(code_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            code = match.group(1).strip()
            # Remove any trailing PAUSE that might have been captured
            if code.endswith('PAUSE'):
                code = code[:-5].strip()
            return ("execute_code", code)
        
        return None
    
    def execute_action(self, action_type: str, action_input: str) -> str:
        """Execute the specified action"""
        if action_type == "web_search":
            return self.web_search.search(action_input)
        elif action_type == "execute_code":
            return self.code_executor.execute(action_input)
        else:
            return f"Unknown action type: {action_type}"
    
    def is_final_answer(self, text: str) -> bool:
        """Check if the response contains a final answer"""
        return "Final Answer:" in text
    
    def extract_final_answer(self, text: str) -> str:
        """Extract the final answer from the response"""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text
    
    def run(self, query: str) -> str:
        """Run the ReAct loop for the given query"""
        print("\n" + "="*80)
        print(f"🤖 AGENT STARTED")
        print(f"📝 User Query: {query}")
        print("="*80 + "\n")
        
        # Reset and add user query
        self.reset()
        self.add_message("user", query)
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'─'*80}")
            print(f"🔄 ITERATION {iteration}")
            print(f"{'─'*80}\n")
            
            # Get response from LLM
            print("💭 Agent is thinking...")
            if self.stream:
                print("\n📋 Agent's Response:")
                print("-" * 80)
                response = self.call_llm()
                print()  # New line after streaming
                print("-" * 80)
            else:
                response = self.call_llm()
                # Display the agent's thought process
                print("\n📋 Agent's Response:")
                print("-" * 80)
                print(response)
                print("-" * 80)
            
            self.add_message("assistant", response)
            
            # IMPORTANT: Check for actions FIRST before checking final answer
            # This ensures tools are executed even if the LLM includes a final answer in the same response
            action = self.parse_action(response)
            
            if action:
                action_type, action_input = action
                print(f"\n⚡ Executing Action: {action_type}")
                print(f"📥 Input: {action_input}\n")
                
                # Execute the action
                observation = self.execute_action(action_type, action_input)
                
                # Display observation
                print("\n👁️ Observation Received:")
                print("-" * 80)
                print(observation[:500] + "..." if len(observation) > 500 else observation)
                print("-" * 80)
                
                # Add observation to messages
                observation_text = f"Observation from {action_type}: {observation}"
                self.add_message("user", observation_text)
                
                # Continue the loop - don't check for final answer yet since we just executed an action
                continue
            
            # Only check for final answer if no actions were found
            if self.is_final_answer(response):
                final_answer = self.extract_final_answer(response)
                print("\n" + "="*80)
                print("✅ AGENT COMPLETED")
                print("="*80)
                print(f"\n🎯 Final Answer:\n{final_answer}\n")
                return final_answer
            
            # No action found and no final answer - might be stuck
            if "PAUSE" not in response:
                print("\n⚠️  No action found and no PAUSE detected. Prompting agent to continue...")
                self.add_message("user", "Please continue with a Thought and Action, or provide a Final Answer.")
        
        # Max iterations reached
        print("\n⚠️  Maximum iterations reached. Generating final answer...")
        self.add_message("user", "Please provide a Final Answer based on the information gathered so far.")
        
        if self.stream:
            print("\n📋 Agent's Final Response:")
            print("-" * 80)
            response = self.call_llm()
            print()  # New line after streaming
            print("-" * 80)
        else:
            response = self.call_llm()
            print("\n📋 Agent's Final Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
        
        final_answer = self.extract_final_answer(response)
        
        print("\n" + "="*80)
        print("✅ AGENT COMPLETED (Max Iterations)")
        print("="*80)
        print(f"\n🎯 Final Answer:\n{final_answer}\n")
        
        return final_answer

