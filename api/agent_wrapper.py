"""API wrapper for ReActAgent that returns structured data"""

from agent.react_agent import ReActAgent as BaseReActAgent
from typing import Dict, List, Any


class ReActAgentAPI(BaseReActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = False  # Disable streaming for API
        self.collected_actions: List[Dict[str, Any]] = []
        self.collected_observations: List[str] = []
    
    def run(self, query: str) -> dict:
        """Modified run to return structured data"""
        # Reset collections
        self.collected_actions = []
        self.collected_observations = []
        
        # Call parent run method but capture data
        print("\n" + "="*80)
        print(f"🤖 AGENT STARTED")
        print(f"📝 User Query: {query}")
        print("="*80 + "\n")
        
        # Reset and add user query
        self.reset()
        self.add_message("user", query)
        
        iteration = 0
        final_answer = None
        status = "completed"
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'─'*80}")
            print(f"🔄 ITERATION {iteration}")
            print(f"{'─'*80}\n")
            
            # Get response from LLM
            print("💭 Agent is thinking...")
            response = self.call_llm()
            print("\n📋 Agent's Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
            self.add_message("assistant", response)
            
            # Check for actions FIRST before checking final answer
            action = self.parse_action(response)
            
            if action:
                action_type, action_input = action
                print(f"\n⚡ Executing Action: {action_type}")
                print(f"📥 Input: {action_input}\n")
                
                # Execute the action
                observation = self.execute_action(action_type, action_input)
                
                # Store action and observation
                self.collected_actions.append({
                    "type": action_type,
                    "input": action_input,
                    "observation": observation
                })
                self.collected_observations.append(observation)
                
                # Display observation
                print("\n👁️ Observation Received:")
                print("-" * 80)
                print(observation[:500] + "..." if len(observation) > 500 else observation)
                print("-" * 80)
                
                # Add observation to messages
                observation_text = f"Observation from {action_type}: {observation}"
                self.add_message("user", observation_text)
                
                # Continue the loop
                continue
            
            # Only check for final answer if no actions were found
            if self.is_final_answer(response):
                final_answer = self.extract_final_answer(response)
                print("\n" + "="*80)
                print("✅ AGENT COMPLETED")
                print("="*80)
                print(f"\n🎯 Final Answer:\n{final_answer}\n")
                break
            
            # No action found and no final answer - might be stuck
            if "PAUSE" not in response:
                print("\n⚠️  No action found and no PAUSE detected. Prompting agent to continue...")
                self.add_message("user", "Please continue with a Thought and Action, or provide a Final Answer.")
        
        # Max iterations reached
        if iteration >= self.max_iterations and not final_answer:
            print("\n⚠️  Maximum iterations reached. Generating final answer...")
            self.add_message("user", "Please provide a Final Answer based on the information gathered so far.")
            
            response = self.call_llm()
            print("\n📋 Agent's Final Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
            final_answer = self.extract_final_answer(response)
            status = "max_iterations_reached"
            
            print("\n" + "="*80)
            print("✅ AGENT COMPLETED (Max Iterations)")
            print("="*80)
            print(f"\n🎯 Final Answer:\n{final_answer}\n")
        
        return {
            "answer": final_answer or "No answer generated",
            "iterations": iteration,
            "actions": self.collected_actions,
            "observations": self.collected_observations,
            "status": status
        }

