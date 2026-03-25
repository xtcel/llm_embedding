"""
FastAPI application for Qwen3-embedding-0.6B service
"""
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.embedder import EmbeddingModel
from src.models import (
    EmbeddingRequest,
    EmbeddingsRequest,
    EmbeddingResponse,
    EmbeddingsResponse,
    HealthResponse,
    ModelInfo,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingObject,
    OpenAIEmbeddingUsage,
    OpenAIEmbeddingResponse,
    OpenAIModelObject,
    OpenAIModelsResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global model instance
embedding_model: Optional[EmbeddingModel] = None

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
if MODEL_NAME.startswith("~"):
    MODEL_NAME = os.path.expanduser(MODEL_NAME)

# Local model path: if set and the directory exists, load from disk instead of downloading
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "").strip()
if LOCAL_MODEL_PATH.startswith("~"):
    LOCAL_MODEL_PATH = os.path.expanduser(LOCAL_MODEL_PATH)
LOCAL_MODEL_PATH = LOCAL_MODEL_PATH or None  # normalize empty string → None

DEVICE = os.getenv("DEVICE", None)  # Auto-detect if None
CACHE_FOLDER = os.getenv("HF_HOME", None)  # HuggingFace cache folder
if CACHE_FOLDER and CACHE_FOLDER.startswith("~"):
    CACHE_FOLDER = os.path.expanduser(CACHE_FOLDER)

NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global embedding_model

    # Startup
    logger.info("Starting Qwen Embedding Service...")
    logger.info(f"Local model path: {LOCAL_MODEL_PATH or '(not set, will download)'}")
    logger.info(f"Model (HuggingFace fallback): {MODEL_NAME}")
    logger.info(f"Device: {DEVICE or 'auto'}")
    logger.info(f"Cache folder: {CACHE_FOLDER or 'default'}")

    # Initialize model
    embedding_model = EmbeddingModel(
        model_name=MODEL_NAME,
        device=DEVICE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        cache_folder=CACHE_FOLDER,
        local_model_path=LOCAL_MODEL_PATH,
    )

    # Pre-load model
    logger.info("Loading model (this may take a few minutes on first run)...")
    _ = embedding_model.model
    logger.info("Model loaded successfully!")

    yield

    # Shutdown
    logger.info("Shutting down Qwen Embedding Service...")
    if embedding_model:
        embedding_model.unload_model()


# Create FastAPI app
app = FastAPI(
    title="Qwen3-Embedding API",
    description="REST API for Qwen3-Embedding-0.6B text embeddings using Sentence Transformers",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the service and model are healthy"""
    if embedding_model is None:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            model_name=None,
            device=None,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=embedding_model.model_name,
        device=embedding_model.device,
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the current model"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    return ModelInfo(
        model_name=embedding_model.model_name,
        embedding_dim=embedding_model.get_embedding_dim(),
        max_seq_length=embedding_model.get_max_seq_length(),
        device=embedding_model.device,
        normalize_embeddings=embedding_model.normalize_embeddings,
    )


@app.post("/embed", response_model=EmbeddingResponse, tags=["Embedding"])
async def embed_text(request: EmbeddingRequest):
    """
    Generate embedding for a single text.

    - **input**: Text to embed (required)
    - **model**: Optional model override
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        start_time = time.time()
        embedding = embedding_model.encode(request.input)
        processing_time = time.time() - start_time

        logger.info(f"Embedded text in {processing_time:.3f}s")

        return EmbeddingResponse(
            object="embedding",
            embedding=embedding,
            index=0,
            model=embedding_model.model_name,
        )

    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/embeddings", response_model=EmbeddingsResponse, tags=["Embedding"])
async def embed_texts(request: EmbeddingsRequest):
    """
    Generate embeddings for multiple texts in batch.

    - **inputs**: List of texts to embed (required)
    - **model**: Optional model override
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        start_time = time.time()
        embeddings = embedding_model.encode(
            request.inputs,
            batch_size=BATCH_SIZE,
        )
        processing_time = time.time() - start_time

        # Build response
        data = [
            EmbeddingResponse(
                object="embedding",
                embedding=emb,
                index=i,
                model=embedding_model.model_name,
            )
            for i, emb in enumerate(embeddings)
        ]

        # Estimate token usage (rough approximation: ~1.3 tokens per word)
        total_chars = sum(len(text) for text in request.inputs)
        total_tokens = int(total_chars / 4)  # Rough approximation

        logger.info(f"Embedded {len(request.inputs)} texts in {processing_time:.3f}s")

        return EmbeddingsResponse(
            object="list",
            data=data,
            model=embedding_model.model_name,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
                "characters": total_chars,
            },
        )

    except Exception as e:
        logger.error(f"Error embedding texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")


@app.post("/similarity", tags=["Utility"])
async def compute_similarity(
    text1: str,
    text2: str,
):
    """
    Compute cosine similarity between two texts.

    - **text1**: First text
    - **text2**: Second text
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        emb1 = embedding_model.encode(text1)
        emb2 = embedding_model.encode(text2)
        similarity = embedding_model.compute_similarity(emb1, emb2)

        return {
            "similarity": similarity,
            "text1_length": len(text1),
            "text2_length": len(text2),
        }

    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")


@app.post("/encode-query", tags=["Embedding"])
async def encode_query(query: str):
    """
    Encode a query for similarity search.
    This is optimized for short query texts.

    - **query**: Query text to encode
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        embedding = embedding_model.encode_query(query)

        return {
            "embedding": embedding,
            "model": embedding_model.model_name,
            "dimension": len(embedding),
        }

    except Exception as e:
        logger.error(f"Error encoding query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query encoding failed: {str(e)}")


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models", response_model=OpenAIModelsResponse, tags=["OpenAI"])
async def openai_list_models():
    """OpenAI-compatible model listing endpoint."""
    import time as _time
    model_id = embedding_model.model_name if embedding_model else MODEL_NAME
    return OpenAIModelsResponse(
        object="list",
        data=[
            OpenAIModelObject(
                id=model_id,
                object="model",
                created=int(_time.time()),
                owned_by="local",
            )
        ],
    )


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse, tags=["OpenAI"])
async def openai_embeddings(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Accepts a single string or an array of strings in **input**.
    Returns embeddings in the same format as the OpenAI Embeddings API.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if request.encoding_format != "float":
        raise HTTPException(
            status_code=400,
            detail=f"encoding_format '{request.encoding_format}' is not supported. Only 'float' is supported.",
        )

    texts = request.input if isinstance(request.input, list) else [request.input]
    if not texts or any(not t for t in texts):
        raise HTTPException(status_code=400, detail="'input' must not be empty.")

    try:
        embeddings = embedding_model.encode(texts, batch_size=BATCH_SIZE)
        # encode() returns List[float] for a single text; wrap it
        if isinstance(embeddings[0], float):
            embeddings = [embeddings]

        total_chars = sum(len(t) for t in texts)
        total_tokens = max(1, int(total_chars / 4))

        return OpenAIEmbeddingResponse(
            object="list",
            data=[
                OpenAIEmbeddingObject(object="embedding", embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=embedding_model.model_name,
            usage=OpenAIEmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in /v1/embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Qwen3-Embedding API",
        "version": "0.1.0",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "embed": "/embed",
            "embeddings": "/embeddings",
            "similarity": "/similarity",
            "encode_query": "/encode-query",
            "openai_models": "/v1/models",
            "openai_embeddings": "/v1/embeddings",
            "docs": "/docs",
        },
    }


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app (for testing)"""
    return app


if __name__ == "__main__":
    import uvicorn

    # Get host and port from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
