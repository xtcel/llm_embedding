"""
API integration tests (requires running service)
"""
import pytest
from httpx import AsyncClient, ASGITransport
from src.app import app


@pytest.fixture
async def client():
    """Create test client"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestRootEndpoint:
    """Tests for root endpoint"""

    @pytest.mark.asyncio
    async def test_root_returns_api_info(self, client):
        """Test root endpoint returns API information"""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Qwen3-Embedding API"
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    @pytest.mark.asyncio
    async def test_health_endpoint_exists(self, client):
        """Test health endpoint exists"""
        response = await client.get("/health")
        # May return 503 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 503]


class TestModelInfoEndpoint:
    """Tests for model info endpoint"""

    @pytest.mark.asyncio
    async def test_model_info_endpoint_exists(self, client):
        """Test model info endpoint exists"""
        response = await client.get("/model-info")
        # May return 503 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 503]


class TestEmbedEndpoint:
    """Tests for single embed endpoint"""

    @pytest.mark.asyncio
    async def test_embed_endpoint_exists(self, client):
        """Test embed endpoint exists"""
        response = await client.post(
            "/embed",
            json={"input": "test text"}
        )
        # May return 503 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 422, 503]

    @pytest.mark.asyncio
    async def test_embed_requires_input(self, client):
        """Test embed endpoint requires input"""
        response = await client.post(
            "/embed",
            json={}
        )
        assert response.status_code == 422  # Validation error


class TestEmbeddingsEndpoint:
    """Tests for batch embeddings endpoint"""

    @pytest.mark.asyncio
    async def test_embeddings_endpoint_exists(self, client):
        """Test embeddings endpoint exists"""
        response = await client.post(
            "/embeddings",
            json={"inputs": ["text1", "text2"]}
        )
        # May return 503 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 422, 503]

    @pytest.mark.asyncio
    async def test_embeddings_requires_inputs(self, client):
        """Test embeddings endpoint requires inputs"""
        response = await client.post(
            "/embeddings",
            json={}
        )
        assert response.status_code == 422


class TestSimilarityEndpoint:
    """Tests for similarity endpoint"""

    @pytest.mark.asyncio
    async def test_similarity_endpoint_exists(self, client):
        """Test similarity endpoint exists"""
        response = await client.post(
            "/similarity",
            params={"text1": "hello", "text2": "world"}
        )
        # May return 503 if model not loaded, but endpoint should exist
        assert response.status_code in [200, 503]
