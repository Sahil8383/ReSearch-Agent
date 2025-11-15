"""Error handling middleware"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """Global error handler middleware"""
    try:
        response = await call_next(request)
        return response
    except RequestValidationError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": str(e),
                "status_code": 422,
                "timestamp": str(datetime.utcnow())
            }
        )
    except StarletteHTTPException as e:
        logger.error(f"HTTP error: {e}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": "HTTP Error",
                "message": e.detail,
                "status_code": e.status_code,
                "timestamp": str(datetime.utcnow())
            }
        )
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "status_code": 500,
                "timestamp": str(datetime.utcnow())
            }
        )

