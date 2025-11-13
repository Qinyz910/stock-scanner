"""
Custom exception classes for error handling and mapping.
"""
from typing import Any, Optional, Dict


class AppException(Exception):
    """Base exception for application-specific errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        self.original_error = original_error
        super().__init__(self.message)


class ValidationError(AppException):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=422,
            details=details,
            original_error=original_error,
        )


class UnauthorizedError(AppException):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Unauthorized",
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            code="UNAUTHORIZED",
            message=message,
            status_code=401,
            details=details,
            original_error=original_error,
        )


class ForbiddenError(AppException):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Forbidden",
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            code="FORBIDDEN",
            message=message,
            status_code=403,
            details=details,
            original_error=original_error,
        )


class NotFoundError(AppException):
    """Raised when a resource is not found."""

    def __init__(
        self,
        message: str = "Not Found",
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            code="NOT_FOUND",
            message=message,
            status_code=404,
            details=details,
            original_error=original_error,
        )


class UpstreamError(AppException):
    """Raised when external dependency fails or times out."""

    def __init__(
        self,
        message: str = "External service error",
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            code="UPSTREAM_ERROR",
            message=message,
            status_code=502,
            details=details,
            original_error=original_error,
        )


class InternalError(AppException):
    """Raised for unexpected internal errors."""

    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[Any] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            status_code=500,
            details=details,
            original_error=original_error,
        )
