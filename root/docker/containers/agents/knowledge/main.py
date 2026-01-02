"""
Knowledge Agent for Ivy AI Orchestration System

This service provides RAG (Retrieval Augmented Generation) capabilities by
implementing two-stage Qdrant retrieval with ColQwen2 embeddings.

Architecture:
- FastAPI server with /query endpoint
- Two-stage retrieval: prefetch with pooled vectors, rerank with original
- ColQwen2 model for query embedding generation
- Returns top-k documents with metadata and scores
"""

import os
from typing import List, Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image


# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://vector-db:6333")
QDRANT_COLLECTION_NAME = "RAG_ColQwen2"
MODEL_NAME = "vidore/colqwen2-v1.0"
HF_HOME = os.getenv("HF_HOME", "/app/.cache/huggingface")

# Initialize FastAPI app
app = FastAPI(title="Knowledge Agent", version="0.1.0")

# Global model instances (loaded on startup)
model = None
processor = None
qdrant_client = None


class QueryRequest(BaseModel):
    query: str
    limit: int = 10


class QueryResponse(BaseModel):
    documents: List[Dict[str, Any]]
    query: str


def load_models():
    """Load ColQwen2 model and processor on startup."""
    global model, processor, qdrant_client
    
    print(f"Loading ColQwen2 model: {MODEL_NAME}")
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    # Load ColQwen2 model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        cache_dir=HF_HOME
    ).eval()
    
    processor = ColQwen2Processor.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_HOME
    )
    
    print("Models loaded successfully")


def generate_query_embedding(query: str) -> List[List[float]]:
    """
    Generate ColQwen2 embeddings for a text query.
    
    Args:
        query: Text query string
        
    Returns:
        List of embedding vectors (multivector representation)
    """
    if model is None or processor is None:
        raise RuntimeError("Models not loaded")
    
    with torch.no_grad():
        # Process query text
        processed_query = processor.process_queries([query]).to(model.device)
        
        # Generate embeddings
        query_embeddings = model(**processed_query)
        
        # Convert to list format for Qdrant
        embedding = query_embeddings[0].cpu().float().numpy().tolist()
    
    return embedding


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base using two-stage retrieval.
    
    Stage 1: Prefetch candidates using mean-pooled vectors (fast HNSW search)
    Stage 2: Rerank candidates using original multivectors (accurate scoring)
    
    Args:
        request: QueryRequest with query text and limit
        
    Returns:
        QueryResponse with ranked documents and metadata
    """
    try:
        # Generate query embedding
        query_embedding = generate_query_embedding(request.query)
        
        # Two-stage retrieval with prefetch
        response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            prefetch=[
                # Stage 1: Fast prefetch using pooled vectors
                models.Prefetch(
                    query=query_embedding,
                    limit=100,  # Retrieve more candidates for reranking
                    using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query_embedding,
                    limit=100,
                    using="mean_pooling_rows"
                ),
            ],
            # Stage 2: Rerank using original multivectors
            limit=request.limit,
            using="original"
        )
        
        # Format results
        documents = []
        for point in response.points:
            documents.append({
                "id": str(point.id),
                "score": point.score,
                "source": point.payload.get("source", "unknown"),
                "page_index": point.payload.get("index", -1),
            })
        
        return QueryResponse(documents=documents, query=request.query)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "qdrant_url": QDRANT_URL,
        "collection": QDRANT_COLLECTION_NAME,
        "models_loaded": model is not None and processor is not None
    }


@app.on_event("startup")
async def startup_event():
    """Load models when the service starts."""
    load_models()


def main():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
