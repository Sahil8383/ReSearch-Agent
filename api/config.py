"""Configuration settings for the API"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://rbmbp33@localhost/first-agent"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False
    
    # API
    API_TITLE: str = "ReAct Agent API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Agent
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_MODEL: str = "claude-3-5-haiku-20241022"
    
    # Session
    MAX_SESSIONS: int = 1000
    SESSION_TIMEOUT: int = 3600  # 1 hour
    
    # Keys
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str
    
    class Config:
        env_file = ".env"


settings = Settings()

