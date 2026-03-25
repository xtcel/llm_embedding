"""
Unit tests for Pydantic models
"""
import pytest
from pydantic import ValidationError
from src.models import (
    EmbeddingRequest,
    EmbeddingsRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelInfo,
)


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest model"""

    def test_valid_single_text(self):
        """Test valid single text input"""
        request = EmbeddingRequest(input="Hello, world!")
        assert request.input == "Hello, world!"
        assert request.model is None

    def test_with_model_name(self):
        """Test with optional model name"""
        request = EmbeddingRequest(
            input="Test text",
            model="custom-model"
        )
        assert request.model == "custom-model"

    def test_empty_input_fails(self):
        """Test that empty input raises validation error"""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="")

    def test_whitespace_only_fails(self):
        """Test that whitespace-only input raises validation error"""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="   ")


class TestEmbeddingsRequest:
    """Tests for EmbeddingsRequest model"""

    def test_valid_list_of_texts(self):
        """Test valid list of texts"""
        request = EmbeddingsRequest(
            inputs=["Hello", "World", "Test"]
        )
        assert len(request.inputs) == 3

    def test_single_item_list(self):
        """Test single item list"""
        request = EmbeddingsRequest(inputs=["Single text"])
        assert len(request.inputs) == 1

    def test_empty_list_fails(self):
        """Test that empty list raises validation error"""
        with pytest.raises(ValidationError):
            EmbeddingsRequest(inputs=[])


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse model"""

    def test_valid_response(self):
        """Test valid embedding response"""
        response = EmbeddingResponse(
            object="embedding",
            embedding=[0.1, 0.2, 0.3],
            index=0,
            model="test-model"
        )
        assert response.object == "embedding"
        assert response.index == 0

    def test_with_float_embeddings(self):
        """Test response with float embedding values"""
        embedding = [float(i) / 100 for i in range(1024)]
        response = EmbeddingResponse(
            object="embedding",
            embedding=embedding,
            index=5,
            model="qwen-embedding"
        )
        assert len(response.embedding) == 1024


class TestHealthResponse:
    """Tests for HealthResponse model"""

    def test_healthy_response(self):
        """Test healthy service response"""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="Qwen3-Embedding-0.6B",
            device="cuda"
        )
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_loading_response(self):
        """Test loading state response"""
        response = HealthResponse(
            status="loading",
            model_loaded=False,
            model_name=None,
            device=None
        )
        assert response.model_loaded is False


class TestModelInfo:
    """Tests for ModelInfo model"""

    def test_valid_model_info(self):
        """Test valid model info"""
        info = ModelInfo(
            model_name="Qwen3-Embedding-0.6B",
            embedding_dim=1024,
            max_seq_length=8192,
            device="cuda",
            normalize_embeddings=True
        )
        assert info.embedding_dim == 1024
        assert info.max_seq_length == 8192
