"""
Unified error response handling and exception mapping.
"""
from typing import Any, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from jose import JWTError
import uuid

from utils.exceptions import (
    AppException,
    ValidationError,
    UnauthorizedError,
    ForbiddenError,
    NotFoundError,
    UpstreamError,
    InternalError,
)
from utils.logger import get_logger

logger = get_logger()


class ErrorResponse(BaseModel):
    """Unified error response model."""

    code: str
    message: str
    details: Optional[Any] = None
    traceId: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": {"field": "email", "reason": "invalid format"},
                "traceId": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def get_trace_id_from_context(request: Request) -> Optional[str]:
    """Extract or generate trace ID from request context."""
    # Try to get trace ID from headers (if OTel or similar is injecting it)
    trace_id = request.headers.get("X-Trace-ID") or request.headers.get("traceparent")
    if not trace_id:
        trace_id = generate_trace_id()
    return trace_id


def create_error_response(
    code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Any] = None,
    trace_id: Optional[str] = None,
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        code=code,
        message=message,
        details=details,
        traceId=trace_id,
    )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application-specific exceptions."""
    trace_id = get_trace_id_from_context(request)
    logger.error(
        f"AppException [{exc.code}] {exc.message}",
        extra={"trace_id": trace_id, "details": exc.details},
    )
    error_response = create_error_response(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    trace_id = get_trace_id_from_context(request)
    
    # Extract validation error details
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error.get("loc", [])[1:]),
            "type": error.get("type"),
            "message": error.get("msg"),
        })
    
    logger.warning(
        f"Validation error in request",
        extra={"trace_id": trace_id, "errors": errors},
    )
    
    error_response = create_error_response(
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"errors": errors},
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(exclude_none=True),
    )


async def jwt_error_handler(request: Request, exc: JWTError) -> JSONResponse:
    """Handle JWT authentication errors."""
    trace_id = get_trace_id_from_context(request)
    logger.warning(
        f"JWT authentication error: {str(exc)}",
        extra={"trace_id": trace_id},
    )
    error_response = create_error_response(
        code="UNAUTHORIZED",
        message="Invalid or expired authentication token",
        details={"error": "jwt_error"},
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=error_response.model_dump(exclude_none=True),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    trace_id = get_trace_id_from_context(request)
    logger.exception(
        f"Unhandled exception",
        extra={"trace_id": trace_id, "exception_type": type(exc).__name__},
    )
    error_response = create_error_response(
        code="INTERNAL_ERROR",
        message="Internal server error",
        details={"error_type": type(exc).__name__},
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(exclude_none=True),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers with the FastAPI app."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(JWTError, jwt_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
