import pytest
from httpx import AsyncClient

from web_server import app


pytestmark = pytest.mark.asyncio


async def test_need_login_endpoint_returns_status_ok() -> None:
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/api/need_login")

    assert response.status_code == 200
    payload = response.json()
    assert "require_login" in payload
    assert isinstance(payload["require_login"], bool)


async def test_config_endpoint_returns_defaults() -> None:
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    assert "announcement" in payload
    assert "default_api_url" in payload
