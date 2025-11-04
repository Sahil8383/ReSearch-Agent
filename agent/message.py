"""Message class for storing conversation history"""


class Message:
    """Simple message class to store conversation history"""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {"role": self.role, "content": self.content}

