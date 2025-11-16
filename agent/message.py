"""Message class for storing conversation history"""

from typing import List, Dict

class Message:
    """Simple message class to store conversation history with short-term memory"""
    
    # Class-level storage for the last 5 messages (short-term memory)
    _short_term_memory: List['Message'] = []
    _memory_size: int = 5
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        
        # Automatically add to short-term memory
        self._add_to_memory()
    
    def _add_to_memory(self):
        """Add this message to short-term memory, keeping only the last N messages"""
        Message._short_term_memory.append(self)
        
        # Keep only the last N messages
        if len(Message._short_term_memory) > Message._memory_size:
            Message._short_term_memory = Message._short_term_memory[-Message._memory_size:]
    
    def to_dict(self):
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def get_short_term_memory(cls) -> List['Message']:
        """Get the last 5 messages from short-term memory"""
        return cls._short_term_memory.copy()
    
    @classmethod
    def get_short_term_memory_dicts(cls) -> List[Dict[str, str]]:
        """Get the last 5 messages as dictionaries for API usage"""
        return [msg.to_dict() for msg in cls._short_term_memory]
    
    @classmethod
    def clear_memory(cls):
        """Clear the short-term memory"""
        cls._short_term_memory = []
    
    @classmethod
    def set_memory_size(cls, size: int):
        """Set the size of short-term memory (default: 5)"""
        cls._memory_size = size
        # Trim memory if new size is smaller
        if len(cls._short_term_memory) > size:
            cls._short_term_memory = cls._short_term_memory[-size:]
    
    @classmethod
    def load_from_db(cls, db_memory: list):
        """Load short-term memory from database format (list of dicts)"""
        cls._short_term_memory = []
        if db_memory:
            for msg_dict in db_memory:
                if isinstance(msg_dict, dict) and "role" in msg_dict and "content" in msg_dict:
                    # Create message without triggering _add_to_memory
                    msg = Message.__new__(Message)
                    msg.role = msg_dict["role"]
                    msg.content = msg_dict["content"]
                    cls._short_term_memory.append(msg)
        # Trim to memory size
        if len(cls._short_term_memory) > cls._memory_size:
            cls._short_term_memory = cls._short_term_memory[-cls._memory_size:]
    
    @classmethod
    def save_to_db(cls) -> list:
        """Save short-term memory to database format (list of dicts)"""
        return [msg.to_dict() for msg in cls._short_term_memory]

