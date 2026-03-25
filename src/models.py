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


# ---------------------------------------------------------------------------
# OpenAI-compatible models  (POST /v1/embeddings)
# ---------------------------------------------------------------------------

class OpenAIEmbeddingRequest(BaseModel):
    """OpenAI-compatible embeddings request."""
    input: Union[str, List[str]] = Field(
        ..., description="Text(s) to embed — a single string or an array of strings."
    )
    model: str = Field(
        "text-embedding-ada-002",
        description="Model ID to use. Ignored at runtime; the server uses its configured model.",
    )
    encoding_format: str = Field(
        "float",
        description="Format for the returned embeddings. Only 'float' is supported.",
    )


class OpenAIEmbeddingObject(BaseModel):
    """A single embedding object in the OpenAI response."""
    object: str = "embedding"
    embedding: List[float] = Field(..., description="Embedding vector.")
    index: int = Field(..., description="Index of the input text.")


class OpenAIEmbeddingUsage(BaseModel):
    """Token usage reported in the OpenAI response."""
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingResponse(BaseModel):
    """OpenAI-compatible embeddings response."""
    object: str = "list"
    data: List[OpenAIEmbeddingObject]
    model: str
    usage: OpenAIEmbeddingUsage


class OpenAIModelObject(BaseModel):
    """A single model entry for the OpenAI /v1/models response."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class OpenAIModelsResponse(BaseModel):
    """OpenAI-compatible /v1/models response."""
    object: str = "list"
    data: List[OpenAIModelObject]
