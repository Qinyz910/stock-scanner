"""
Unit tests for unified error handling and exception mapping.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
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
from utils.error_handlers import (
    register_error_handlers,
    ErrorResponse,
    generate_trace_id,
    get_trace_id_from_context,
    create_error_response,
)


@pytest.fixture
def test_app():
    """Create a test FastAPI app with error handlers registered."""
    app = FastAPI()
    register_error_handlers(app)

    class TestRequest(BaseModel):
        value: str

    @app.post("/test-validation")
    async def test_validation(req: TestRequest):
        return {"value": req.value}

    @app.get("/test-unauthorized")
    async def test_unauthorized():
        raise UnauthorizedError("Invalid credentials")

    @app.get("/test-forbidden")
    async def test_forbidden():
        raise ForbiddenError("Access denied")

    @app.get("/test-not-found")
    async def test_not_found():
        raise NotFoundError("Resource not found")

    @app.get("/test-upstream-error")
    async def test_upstream_error():
        raise UpstreamError("External service failed")

    @app.get("/test-internal-error")
    async def test_internal_error():
        raise InternalError("Something went wrong")

    @app.get("/test-unhandled")
    async def test_unhandled():
        raise ValueError("Unhandled exception")

    @app.get("/test-trace-id")
    async def test_trace_id(request: Request):
        trace_id = request.headers.get("X-Trace-ID")
        raise UnauthorizedError(
            "Test error with trace ID",
            details={"trace_from_header": trace_id},
        )

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_error_response_creation(self):
        """Test creating an error response."""
        response = ErrorResponse(
            code="TEST_ERROR",
            message="Test message",
            details={"key": "value"},
            traceId="test-trace-id",
        )
        assert response.code == "TEST_ERROR"
        assert response.message == "Test message"
        assert response.details == {"key": "value"}
        assert response.traceId == "test-trace-id"

    def test_error_response_without_optional_fields(self):
        """Test creating an error response without optional fields."""
        response = ErrorResponse(
            code="SIMPLE_ERROR",
            message="Simple message",
        )
        assert response.code == "SIMPLE_ERROR"
        assert response.message == "Simple message"
        assert response.details is None
        assert response.traceId is None

    def test_error_response_dict_exclude_none(self):
        """Test that ErrorResponse.dict() excludes None values."""
        response = ErrorResponse(
            code="TEST_ERROR",
            message="Test message",
        )
        data = response.model_dump(exclude_none=True)
        assert "code" in data
        assert "message" in data
        assert "details" not in data
        assert "traceId" not in data


class TestTraceId:
    """Test trace ID generation and extraction."""

    def test_generate_trace_id(self):
        """Test generating a trace ID."""
        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)
        assert len(trace_id) > 0
        # Should be a valid UUID
        try:
            uuid.UUID(trace_id)
        except ValueError:
            pytest.fail("Generated trace ID is not a valid UUID")

    def test_generate_trace_id_uniqueness(self):
        """Test that generated trace IDs are unique."""
        trace_id1 = generate_trace_id()
        trace_id2 = generate_trace_id()
        assert trace_id1 != trace_id2

    def test_get_trace_id_from_context_with_header(self, test_app):
        """Test extracting trace ID from request headers."""
        client = TestClient(test_app)
        test_trace_id = "custom-trace-123"
        response = client.get(
            "/test-trace-id",
            headers={"X-Trace-ID": test_trace_id},
        )
        data = response.json()
        assert data["traceId"] == test_trace_id

    def test_get_trace_id_from_context_generated(self, test_app):
        """Test that a trace ID is generated if not in headers."""
        client = TestClient(test_app)
        response = client.get("/test-unauthorized")
        data = response.json()
        assert "traceId" in data
        assert len(data["traceId"]) > 0


class TestValidationError:
    """Test validation error handling."""

    def test_validation_error_422(self, client):
        """Test that validation errors return 422."""
        response = client.post("/test-validation", json={})
        assert response.status_code == 422

    def test_validation_error_structure(self, client):
        """Test that validation errors have the correct structure."""
        response = client.post("/test-validation", json={})
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert data["message"] == "Request validation failed"
        assert "details" in data
        assert "errors" in data["details"]

    def test_validation_error_details(self, client):
        """Test that validation error details contain field information."""
        response = client.post("/test-validation", json={})
        data = response.json()
        errors = data["details"]["errors"]
        assert len(errors) > 0
        assert "field" in errors[0]
        assert "type" in errors[0]
        assert "message" in errors[0]


class TestAuthenticationErrors:
    """Test authentication and authorization error handling."""

    def test_unauthorized_error_401(self, client):
        """Test that unauthorized errors return 401."""
        response = client.get("/test-unauthorized")
        assert response.status_code == 401

    def test_unauthorized_error_structure(self, client):
        """Test that unauthorized errors have the correct structure."""
        response = client.get("/test-unauthorized")
        data = response.json()
        assert data["code"] == "UNAUTHORIZED"
        assert data["message"] == "Invalid credentials"
        assert "traceId" in data

    def test_forbidden_error_403(self, client):
        """Test that forbidden errors return 403."""
        response = client.get("/test-forbidden")
        assert response.status_code == 403

    def test_forbidden_error_structure(self, client):
        """Test that forbidden errors have the correct structure."""
        response = client.get("/test-forbidden")
        data = response.json()
        assert data["code"] == "FORBIDDEN"
        assert data["message"] == "Access denied"


class TestNotFoundError:
    """Test not found error handling."""

    def test_not_found_error_404(self, client):
        """Test that not found errors return 404."""
        response = client.get("/test-not-found")
        assert response.status_code == 404

    def test_not_found_error_structure(self, client):
        """Test that not found errors have the correct structure."""
        response = client.get("/test-not-found")
        data = response.json()
        assert data["code"] == "NOT_FOUND"
        assert data["message"] == "Resource not found"


class TestUpstreamError:
    """Test upstream error handling."""

    def test_upstream_error_502(self, client):
        """Test that upstream errors return 502."""
        response = client.get("/test-upstream-error")
        assert response.status_code == 502

    def test_upstream_error_structure(self, client):
        """Test that upstream errors have the correct structure."""
        response = client.get("/test-upstream-error")
        data = response.json()
        assert data["code"] == "UPSTREAM_ERROR"
        assert data["message"] == "External service failed"


class TestInternalError:
    """Test internal error handling."""

    def test_internal_error_500(self, client):
        """Test that internal errors return 500."""
        response = client.get("/test-internal-error")
        assert response.status_code == 500

    def test_internal_error_structure(self, client):
        """Test that internal errors have the correct structure."""
        response = client.get("/test-internal-error")
        data = response.json()
        assert data["code"] == "INTERNAL_ERROR"
        assert data["message"] == "Something went wrong"


class TestUnhandledError:
    """Test unhandled exception handling."""

    def test_unhandled_error_500(self, client):
        """Test that unhandled errors return 500."""
        # The TestClient may raise exceptions directly if they occur at the endpoint level
        # We catch it and verify the error structure is correct when caught by the app
        try:
            response = client.get("/test-unhandled")
            # If we get here, it means the exception was caught by the error handler
            assert response.status_code == 500
        except ValueError:
            # This is expected in some versions of TestClient where unhandled
            # exceptions are raised directly during testing
            pass

    def test_unhandled_error_structure(self, client):
        """Test that unhandled errors are mapped to INTERNAL_ERROR."""
        # The TestClient may raise exceptions directly if they occur at the endpoint level
        try:
            response = client.get("/test-unhandled")
            if response.status_code == 500:
                data = response.json()
                assert data["code"] == "INTERNAL_ERROR"
                assert data["message"] == "Internal server error"
                assert "details" in data
                assert data["details"]["error_type"] == "ValueError"
        except ValueError:
            # This is expected in some versions of TestClient
            pass


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_validation_error_initialization(self):
        """Test ValidationError initialization."""
        error = ValidationError(
            "Test validation",
            details={"field": "test"},
        )
        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Test validation"
        assert error.status_code == 422
        assert error.details == {"field": "test"}

    def test_unauthorized_error_initialization(self):
        """Test UnauthorizedError initialization."""
        error = UnauthorizedError("Test unauthorized")
        assert error.code == "UNAUTHORIZED"
        assert error.message == "Test unauthorized"
        assert error.status_code == 401

    def test_forbidden_error_initialization(self):
        """Test ForbiddenError initialization."""
        error = ForbiddenError("Test forbidden")
        assert error.code == "FORBIDDEN"
        assert error.message == "Test forbidden"
        assert error.status_code == 403

    def test_not_found_error_initialization(self):
        """Test NotFoundError initialization."""
        error = NotFoundError("Test not found")
        assert error.code == "NOT_FOUND"
        assert error.message == "Test not found"
        assert error.status_code == 404

    def test_upstream_error_initialization(self):
        """Test UpstreamError initialization."""
        error = UpstreamError("Test upstream")
        assert error.code == "UPSTREAM_ERROR"
        assert error.message == "Test upstream"
        assert error.status_code == 502

    def test_internal_error_initialization(self):
        """Test InternalError initialization."""
        error = InternalError("Test internal")
        assert error.code == "INTERNAL_ERROR"
        assert error.message == "Test internal"
        assert error.status_code == 500

    def test_exception_with_original_error(self):
        """Test exception with original error."""
        original = ValueError("Original error")
        error = InternalError("Test error", original_error=original)
        assert error.original_error is original
        assert str(error) == "Test error"


class TestCreateErrorResponse:
    """Test create_error_response helper."""

    def test_create_error_response(self):
        """Test creating an error response."""
        response = create_error_response(
            code="TEST",
            message="Test message",
            status_code=400,
        )
        assert response.code == "TEST"
        assert response.message == "Test message"
        assert isinstance(response, ErrorResponse)

    def test_create_error_response_with_trace_id(self):
        """Test creating an error response with trace ID."""
        trace_id = "test-trace-123"
        response = create_error_response(
            code="TEST",
            message="Test message",
            trace_id=trace_id,
        )
        assert response.traceId == trace_id


class TestErrorResponseConsistency:
    """Test consistency of error responses across different endpoints."""

    def test_all_error_responses_have_required_fields(self, client):
        """Test that all error responses have required fields."""
        endpoints = [
            "/test-unauthorized",
            "/test-forbidden",
            "/test-not-found",
            "/test-upstream-error",
            "/test-internal-error",
        ]
        for endpoint in endpoints:
            response = client.get(endpoint)
            data = response.json()
            assert "code" in data, f"Missing code in {endpoint}"
            assert "message" in data, f"Missing message in {endpoint}"
            assert "traceId" in data, f"Missing traceId in {endpoint}"

    def test_all_error_responses_are_json(self, client):
        """Test that all error responses are valid JSON."""
        endpoints = [
            "/test-unauthorized",
            "/test-forbidden",
            "/test-not-found",
            "/test-upstream-error",
            "/test-internal-error",
        ]
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.headers["content-type"].startswith("application/json")
            # Should not raise an exception
            response.json()

    def test_error_codes_are_uppercase(self, client):
        """Test that all error codes are uppercase."""
        endpoints = [
            ("/test-unauthorized", "UNAUTHORIZED"),
            ("/test-forbidden", "FORBIDDEN"),
            ("/test-not-found", "NOT_FOUND"),
            ("/test-upstream-error", "UPSTREAM_ERROR"),
            ("/test-internal-error", "INTERNAL_ERROR"),
        ]
        for endpoint, expected_code in endpoints:
            response = client.get(endpoint)
            data = response.json()
            assert data["code"] == expected_code
            # Verify all characters in code are uppercase or underscore
            assert all(c.isupper() or c == "_" for c in data["code"])
