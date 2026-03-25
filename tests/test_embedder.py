"""
Unit tests for EmbeddingModel
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.embedder import EmbeddingModel


class TestEmbeddingModelInit:
    """Tests for EmbeddingModel initialization"""

    def test_default_initialization(self):
        """Test default initialization parameters"""
        model = EmbeddingModel()
        assert model.model_name == "Qwen/Qwen3-Embedding-0.6B"
        assert model.normalize_embeddings is True

    def test_custom_model_name(self):
        """Test custom model name"""
        model = EmbeddingModel(model_name="custom/model")
        assert model.model_name == "custom/model"

    def test_normalize_embeddings_setting(self):
        """Test normalize embeddings setting"""
        model_no_normalize = EmbeddingModel(normalize_embeddings=False)
        assert model_no_normalize.normalize_embeddings is False

        model_normalize = EmbeddingModel(normalize_embeddings=True)
        assert model_normalize.normalize_embeddings is True

    def test_device_auto_detection(self):
        """Test device is set (cpu as fallback for testing)"""
        model = EmbeddingModel(device="cpu")
        assert model.device == "cpu"


class TestEmbeddingModelLazyLoading:
    """Tests for lazy loading behavior"""

    def test_model_not_loaded_initially(self):
        """Test that model is not loaded on initialization"""
        model = EmbeddingModel()
        assert model._model is None

    @patch('src.embedder.SentenceTransformer')
    def test_model_loaded_on_first_access(self, mock_st):
        """Test that model is loaded on first property access"""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_instance

        model = EmbeddingModel()
        _ = model.model  # Access the property

        mock_st.assert_called_once()


class TestEmbeddingModelEncode:
    """Tests for encode functionality"""

    @patch('src.embedder.SentenceTransformer')
    def test_encode_single_text(self, mock_st):
        """Test encoding single text"""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_instance

        model = EmbeddingModel()
        model._model = mock_instance

        result = model.encode("Hello world")
        assert isinstance(result, list)
        assert len(result) == 3

    @patch('src.embedder.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_st):
        """Test encoding multiple texts"""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_st.return_value = mock_instance

        model = EmbeddingModel()
        model._model = mock_instance

        result = model.encode(["Hello", "World"])
        assert len(result) == 2


class TestEmbeddingModelSimilarity:
    """Tests for similarity computation"""

    def test_compute_similarity_normalized(self):
        """Test cosine similarity for normalized vectors"""
        model = EmbeddingModel()

        emb1 = [1.0, 0.0]
        emb2 = [1.0, 0.0]

        similarity = model.compute_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 1e-6

    def test_compute_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors"""
        model = EmbeddingModel()

        emb1 = [1.0, 0.0]
        emb2 = [0.0, 1.0]

        similarity = model.compute_similarity(emb1, emb2)
        assert abs(similarity) < 1e-6

    def test_compute_similarity_opposite(self):
        """Test cosine similarity for opposite vectors"""
        model = EmbeddingModel()

        emb1 = [1.0, 0.0]
        emb2 = [-1.0, 0.0]

        similarity = model.compute_similarity(emb1, emb2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_compute_similarity_zero_vector(self):
        """Test similarity with zero vector"""
        model = EmbeddingModel()

        emb1 = [0.0, 0.0]
        emb2 = [1.0, 0.0]

        similarity = model.compute_similarity(emb1, emb2)
        assert similarity == 0.0


class TestEmbeddingModelUnload:
    """Tests for model unloading"""

    @patch('src.embedder.SentenceTransformer')
    def test_unload_model(self, mock_st):
        """Test unloading model from memory"""
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        model = EmbeddingModel()
        model._model = mock_instance

        model.unload_model()

        assert model._model is None

    @patch('torch.cuda')
    @patch('src.embedder.SentenceTransformer')
    def test_unload_clears_cuda_cache(self, mock_st, mock_cuda):
        """Test that unloading clears CUDA cache"""
        mock_cuda.is_available.return_value = True
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        model = EmbeddingModel(device="cuda")
        model._model = mock_instance

        model.unload_model()

        mock_cuda.empty_cache.assert_called_once()
