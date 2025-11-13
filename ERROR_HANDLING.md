# Unified Error Response and Exception Mapping

## Overview

This document describes the unified error response system implemented in the Stock Scanner API. This system provides:

- **Consistent error response structure** across all API endpoints
- **Standardized HTTP status codes** mapped to specific error types
- **Machine-readable error codes** for frontend error handling
- **Optional trace IDs** for error tracking and debugging

## Error Response Structure

All error responses follow a consistent JSON format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    "additional": "error details"
  },
  "traceId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Fields

- **code** (required): Machine-readable error code (uppercase with underscores)
- **message** (required): Human-readable error message
- **details** (optional): Additional error context (object)
- **traceId** (optional): Unique identifier for tracking/debugging

## Error Types and HTTP Status Codes

### VALIDATION_ERROR (422)

Raised when request validation fails, including:
- Missing required fields
- Invalid field types
- Invalid field formats
- Business logic validation failures

**Example:**
```json
{
  "code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": {
    "errors": [
      {
        "field": "symbols",
        "type": "value_error",
        "message": "field required"
      }
    ]
  },
  "traceId": "..."
}
```

### UNAUTHORIZED (401)

Raised when authentication fails, including:
- Invalid or missing JWT token
- Expired token
- Invalid credentials

**Triggered by:**
- Failed login attempts
- Missing or invalid authentication headers
- JWT validation failures

### FORBIDDEN (403)

Raised when user lacks required permissions.

**Triggered by:**
- Access control failures
- Insufficient permissions
- Role-based access restrictions

### NOT_FOUND (404)

Raised when requested resource is not found.

**Triggered by:**
- Non-existent resource ID
- Invalid resource references
- Deleted resources

### UPSTREAM_ERROR (502)

Raised when external dependencies fail or time out.

**Triggered by:**
- External API failures
- Network timeouts
- Connection errors
- Third-party service errors

**Example:**
```json
{
  "code": "UPSTREAM_ERROR",
  "message": "External service error",
  "details": {
    "status_code": 503,
    "service": "ai_provider"
  },
  "traceId": "..."
}
```

### INTERNAL_ERROR (500)

Raised for unexpected internal server errors.

**Triggered by:**
- Unhandled exceptions
- Database errors
- Configuration errors
- Unexpected runtime errors

## Custom Exception Classes

The system provides custom exception classes in `utils/exceptions.py`:

```python
from utils.exceptions import (
    ValidationError,
    UnauthorizedError,
    ForbiddenError,
    NotFoundError,
    UpstreamError,
    InternalError,
)

# Usage
raise ValidationError(
    "Invalid stock symbol",
    details={"symbol": "INVALID"}
)

raise NotFoundError("Resource not found")

raise UpstreamError(
    "AI service unavailable",
    original_error=external_exception
)
```

## Error Handlers Registration

Error handlers are automatically registered when the FastAPI app is created:

```python
from utils.error_handlers import register_error_handlers

app = FastAPI()
register_error_handlers(app)
```

## Trace ID Support

Trace IDs are automatically generated and included in error responses for debugging:

```python
# Generated UUID format: 550e8400-e29b-41d4-a716-446655440000
{
  "code": "VALIDATION_ERROR",
  "message": "...",
  "traceId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### OpenTelemetry Integration

If OpenTelemetry is configured, trace IDs can be extracted from incoming headers:
- `X-Trace-ID` header
- `traceparent` header (W3C Trace Context)

## OpenAPI Documentation

Error response schemas are included in OpenAPI documentation under `/docs`:

- `ErrorResponse` - Base error response schema
- `ValidationErrorResponse` - Validation error schema
- `UnauthorizedErrorResponse` - Authentication error schema
- `ForbiddenErrorResponse` - Authorization error schema
- `NotFoundErrorResponse` - Not found error schema
- `UpstreamErrorResponse` - Upstream error schema
- `InternalErrorResponse` - Internal error schema

## Best Practices

### For Developers

1. **Use specific exception types** instead of generic exceptions:
   ```python
   # Good
   raise ValidationError("Invalid email format")
   
   # Less specific
   raise AppException("Invalid email format", code="VALIDATION_ERROR", status_code=422)
   ```

2. **Include relevant details**:
   ```python
   raise NotFoundError(
       "Stock not found",
       details={"symbol": symbol, "market": market}
   )
   ```

3. **Preserve original exceptions** for debugging:
   ```python
   try:
       # external call
   except Exception as e:
       raise UpstreamError(
           "API call failed",
           original_error=e
       )
   ```

4. **Catch and re-raise custom exceptions**:
   ```python
   try:
       result = await compute_scores(request)
       return result
   except AppException:
       raise  # Re-raise custom exceptions
   except Exception as e:
       raise InternalError(
           "Computation failed",
           original_error=e
       )
   ```

### For Frontend

1. **Check error codes for routing**:
   ```typescript
   if (response.code === 'VALIDATION_ERROR') {
     // Show validation error UI
   } else if (response.code === 'UNAUTHORIZED') {
     // Redirect to login
   } else if (response.code === 'UPSTREAM_ERROR') {
     // Show retry option
   }
   ```

2. **Use trace IDs for error reporting**:
   ```typescript
   console.log(`Error [${response.traceId}]: ${response.message}`);
   // Include traceId in bug reports
   ```

3. **Display user-friendly messages**:
   ```typescript
   const userMessage = response.message || 'An error occurred';
   showErrorNotification(userMessage);
   ```

## Testing

Unit tests for error handling are located in `tests/test_error_handling.py`:

```bash
# Run all error handling tests
pytest tests/test_error_handling.py -v

# Run specific test class
pytest tests/test_error_handling.py::TestValidationError -v

# Run with coverage
pytest tests/test_error_handling.py --cov=utils.error_handlers --cov=utils.exceptions
```

## Examples

### Example 1: Validation Error

**Request:**
```bash
POST /api/scores
Content-Type: application/json

{}
```

**Response:**
```json
{
  "code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": {
    "errors": [
      {
        "field": "symbols",
        "type": "value_error",
        "message": "field required"
      },
      {
        "field": "factors",
        "type": "value_error",
        "message": "field required"
      }
    ]
  },
  "traceId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Example 2: Unauthorized Error

**Request:**
```bash
GET /api/check_auth
```

**Response (401):**
```json
{
  "code": "UNAUTHORIZED",
  "message": "Invalid authentication credentials",
  "details": {},
  "traceId": "550e8400-e29b-41d4-a716-446655440001"
}
```

### Example 3: Upstream Error

**Request:**
```bash
POST /api/test_api_connection
Content-Type: application/json

{
  "api_url": "https://api.invalid.com",
  "api_key": "invalid-key"
}
```

**Response (502):**
```json
{
  "code": "UPSTREAM_ERROR",
  "message": "API connection test failed: Unauthorized",
  "details": {
    "status_code": 401
  },
  "traceId": "550e8400-e29b-41d4-a716-446655440002"
}
```

## Migration Guide

If you're migrating from the old error handling system:

### Old Way
```python
raise HTTPException(status_code=400, detail="Invalid input")
```

### New Way
```python
from utils.exceptions import ValidationError
raise ValidationError("Invalid input")
```

### Old Way (500 error)
```python
raise HTTPException(status_code=500, detail="Internal error")
```

### New Way
```python
from utils.exceptions import InternalError
raise InternalError("Internal error")
```

## References

- **Implementation**: `utils/exceptions.py`, `utils/error_handlers.py`
- **Tests**: `tests/test_error_handling.py`
- **Integration**: `web_server.py`, `api_v2.py`
- **OpenAPI Schema**: Generated at `/docs`
