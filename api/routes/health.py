"""Health check endpoint"""

from fastapi import APIRouter
from ..schemas import HealthResponse
from ..config import settings
from datetime import datetime

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        database="connected",
        version=settings.API_VERSION
    )

