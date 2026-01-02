# Knowledge Agent

The Knowledge Agent provides RAG (Retrieval Augmented Generation) capabilities using ColQwen2 embeddings and Qdrant vector database with two-stage retrieval.

## Features

- **Two-stage retrieval**: Fast prefetch with pooled vectors, accurate rerank with original multivectors
- **ColQwen2 embeddings**: Vision-language model for document understanding
- **FastAPI endpoint**: `/query` for document retrieval
- **Automatic GPU detection**: Uses CUDA if available, falls back to CPU
- **Separate model cache**: Independent from ingestion service

## API Endpoints

### POST `/query`

Query the knowledge base for relevant documents.

**Request body:**
```json
{
  "query": "What is the payload fairing design?",
  "limit": 10
}
```

**Response:**
```json
{
  "documents": [
    {
      "id": "uuid-string",
      "score": 0.8542,
      "source": "ECSS-E-ST-32-08C.pdf",
      "page_index": 45
    },
    {
      "id": "uuid-string",
      "score": 0.8234,
      "source": "ECSS-E-ST-32-08C.pdf",
      "page_index": 46
    }
  ],
  "query": "What is the payload fairing design?"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "vidore/colqwen2-v1.0",
  "qdrant_url": "http://vector-db:6333",
  "collection": "RAG_ColQwen2",
  "models_loaded": true
}
```

## Two-Stage Retrieval Architecture

Based on [Qdrant PDF Retrieval at Scale](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/)

### Stage 1: Prefetch (Fast)
- Query against mean-pooled vectors (columns + rows)
- Uses HNSW index for fast approximate search
- Fetches top 100 candidates from each pooling strategy
- ~10-50ms on typical hardware

### Stage 2: Rerank (Accurate)
- Rescore prefetched results using original multivectors
- No HNSW overhead (disabled for original vectors)
- Returns final top-k results
- Provides accurate MaxSim scoring

**Performance characteristics:**
- **Without prefetch**: ~500ms for accurate search (linear scan)
- **With prefetch**: ~50-100ms for accurate results (hybrid approach)
- **Accuracy**: Maintains 95%+ recall vs full linear search

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `QDRANT_URL` | Qdrant service URL | `http://vector-db:6333` | Yes |
| `HF_HOME` | Hugging Face cache directory | `/app/.cache/huggingface` | Yes |

### Model Configuration

The agent uses `vidore/colqwen2-v1.0` model:
- **Architecture**: Vision-language transformer
- **Embedding dimension**: 128 (per patch)
- **Vectors per page**: ~700 (depends on document)
- **Precision**: bfloat16 (GPU) or float32 (CPU)

## Query Embedding Generation

Text queries are embedded using ColQwen2 processor:

```python
# Query: "What is the payload fairing?"
# Output: [128-dim vector 1, 128-dim vector 2, ..., 128-dim vector N]
# Typical N: 10-20 vectors for short queries
```

The multivector representation captures semantic nuances better than single-vector embeddings.

## Dependencies

- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **qdrant-client**: Qdrant vector database client
- **colpali-engine**: ColQwen2 model and processor
- **torch**: PyTorch for model inference
- **pillow**: Image processing (unused in query, required by colpali)
- **python-dotenv**: Environment variable management

## Development

### Running Locally

```bash
cd root/docker/containers/agents/knowledge
uv sync
uv run main.py
```

### Testing the API

```bash
# Health check
curl http://localhost:8001/health

# Query documents
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the payload fairing design?",
    "limit": 5
  }'
```

## Performance Tuning

### Adjust Prefetch Limits

Edit [main.py](main.py) to change prefetch limits:

```python
# Increase for better recall (slower)
prefetch=[
    models.Prefetch(query=query_embedding, limit=200, using="mean_pooling_columns"),
    models.Prefetch(query=query_embedding, limit=200, using="mean_pooling_rows"),
]

# Decrease for faster queries (may miss relevant docs)
prefetch=[
    models.Prefetch(query=query_embedding, limit=50, using="mean_pooling_columns"),
    models.Prefetch(query=query_embedding, limit=50, using="mean_pooling_rows"),
]
```

### GPU Acceleration

The agent automatically uses GPU if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**CPU performance**: ~100-200ms per query
**GPU performance**: ~30-50ms per query

### Model Caching

Models are cached in `data/knowledge-cache/` volume:
- First startup: ~5-10 minutes to download ColQwen2
- Subsequent startups: ~30 seconds to load from cache

## Troubleshooting

### Models not loading

```bash
# Check cache directory
ls -lh data/knowledge-cache/huggingface/hub/

# View logs
docker-compose logs knowledge

# Restart with fresh cache
docker-compose down
rm -rf data/knowledge-cache/
docker-compose up --build knowledge
```

### No results returned

```bash
# Verify Qdrant connection
curl http://localhost:6333/collections/RAG_ColQwen2

# Check document count
curl http://localhost:6333/collections/RAG_ColQwen2/points/count

# Test query with verbose output
docker-compose logs -f knowledge
# Then make a query request
```

### Slow queries

1. Check if GPU is available: `docker-compose logs knowledge | grep "Using device"`
2. Reduce prefetch limits (see Performance Tuning)
3. Ensure Qdrant HNSW indexes are built (automatic after ingestion)

## Integration with Orchestrator

The orchestrator calls the knowledge agent for each user query:

```python
# In orchestrator
documents = await query_knowledge_agent(user_message, limit=5)

# Returns:
# [
#   {"id": "...", "score": 0.85, "source": "doc.pdf", "page_index": 10},
#   ...
# ]
```

Documents are then injected as context into the LLM prompt.

## Future Enhancements

- **Reranking models**: Add cross-encoder for even better accuracy
- **Hybrid search**: Combine semantic search with keyword matching
- **Query expansion**: Use LLM to generate multiple query variants
- **Metadata filtering**: Filter by document type, date, author, etc.
- **Caching**: Cache frequent queries for instant responses
- **Batch queries**: Process multiple queries in parallel
