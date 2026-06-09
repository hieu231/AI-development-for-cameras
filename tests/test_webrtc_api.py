"""
Unit tests for WebRTC signaling API endpoints.

These tests validate the API contract (request/response schemas, error cases)
without actually performing WebRTC negotiation.
"""
import pytest
from uuid import uuid4
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client with mocked database dependencies."""
    # Patch database imports before loading server
    with patch("src.database.get_db.engine"), \
         patch("src.database.get_db.Base"):
        from src.server import app
        return TestClient(app)


class TestWebRTCStatus:
    def test_status_endpoint(self, client):
        resp = client.get("/api/webrtc/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "aiortc_available" in data
        assert "active_connections" in data


class TestListConnections:
    def test_empty_connections(self, client):
        resp = client.get("/api/webrtc/connections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 0
        assert isinstance(data["connections"], list)


class TestOfferEndpoint:
    def test_offer_missing_camera(self, client):
        """POST /offer with a camera that is not running should return 404."""
        resp = client.post("/api/webrtc/offer", json={
            "camera_id": str(uuid4()),
            "sdp": "v=0\r\n",
            "type": "offer",
            "fps": 24,
        })
        # Camera not running → 404
        assert resp.status_code == 404

    def test_offer_invalid_body(self, client):
        """POST /offer with missing fields should return 422."""
        resp = client.post("/api/webrtc/offer", json={})
        assert resp.status_code == 422


class TestCloseConnection:
    def test_close_nonexistent(self, client):
        resp = client.delete(f"/api/webrtc/connections/{uuid4()}")
        assert resp.status_code == 404
