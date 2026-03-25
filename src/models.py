"""
Pydantic models for API request/response
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Single text embedding request"""
    input: str = Field(..., description="Text to embed", min_length=1)
    model: Optional[str] = Field(None, description="Model name (optional, uses default if not specified)")


class EmbeddingsRequest(BaseModel):
    """Batch embeddings request"""
    inputs: List[str] = Field(..., description="List of texts to embed", min_length=1)
    model: Optional[str] = Field(None, description="Model name (optional, uses default if not specified)")


class EmbeddingResponse(BaseModel):
    """Single embedding response"""
    object: str = "embedding"
    embedding: List[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index of the input text")
    model: str = Field(..., description="Model used for embedding")


class EmbeddingsResponse(BaseModel):
    """Batch embeddings response"""
    object: str = "list"
    data: List[EmbeddingResponse] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embedding")
    usage: dict = Field(..., description="Token usage information")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Current model name")
    device: Optional[str] = Field(None, description="Device used for inference")


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str = Field(..., description="Model identifier")
    embedding_dim: int = Field(..., description="Embedding dimension")
    max_seq_length: int = Field(..., description="Maximum sequence length")
    device: str = Field(..., description="Device for inference")
    normalize_embeddings: bool = Field(..., description="Whether embeddings are normalized")
