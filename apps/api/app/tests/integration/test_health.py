from httpx import ASGITransport, AsyncClient

from app.main import create_app


async def test_health_endpoint() -> None:
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "showcasehost-api",
        "version": "0.1.0",
    }
