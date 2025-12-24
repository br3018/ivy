# Ivy - AI Orchestration Service

## Architecture Overview

Ivy is a multi-layered LangGraph-based orchestration system with strict service boundaries:

- **UI Layer** → **Orchestrator Layer** → **Agent Layer** → **Knowledge Layer**
- Each layer communicates through defined interfaces only
- Orchestrator delegates tasks to specialized agents, which query knowledge stores

**Key Services** (all Docker containerized):
- `orchestrator` - LangGraph-based coordinator (LangGraph dependency, currently basic stub)
- `ingestion` - PDF → Qdrant vector store pipeline using ColQwen2 embeddings
- `vector-db` - Qdrant instance (ports 6333, 6334)
- `agents/knowledge` - RAG agent (currently basic stub)
- `ui` - Frontend interface (currently basic stub)

## Critical Implementation Details

### Ingestion Service Architecture
The ingestion service ([root/docker/containers/ingestion/main.py](root/docker/containers/ingestion/main.py)) implements a sophisticated PDF-to-vector pipeline:

**Embedding Strategy**: Uses ColQwen2 with three parallel vector representations:
- `original`: Full multivector embeddings (~700 vectors/page, 128-dim, HNSW disabled for speed)
- `mean_pooling_rows`: Row-wise pooled embeddings (128-dim, HNSW enabled)
- `mean_pooling_columns`: Column-wise pooled embeddings (128-dim, HNSW enabled)
- All use MAX_SIM comparator for multivector search

**Data Flow**:
1. Monitors `data/ingestion/unprocessed/` for PDFs
2. Converts pages to images (200 DPI) via pdf2image
3. Processes in batches (default: batch_size=1 due to compute constraints)
4. Stores in Qdrant collection "RAG_ColQwen2" with metadata (source, index)
5. Moves processed PDFs to `data/ingestion/processed/`

**Critical Pattern**: Image embeddings preserve spatial structure through patch-based encoding. The `get_patches()` function dynamically calculates rows/columns based on model (ColPali: fixed 32x32, ColQwen: dynamic via `spatial_merge_size`).

### Two-Stage Retrieval Architecture
**Based on**: [Qdrant PDF Retrieval at Scale](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/)

RAG queries use a two-stage approach to balance speed and accuracy:

1. **Stage 1 - Prefetch**: Query against mean-pooled vectors (columns + rows)
   - Uses HNSW index for fast retrieval
   - Fetches top 100 candidates from each pooling strategy
   - Example: `prefetch_limit = 100` per vector type

2. **Stage 2 - Rerank**: Rescore prefetched results using original multivectors
   - No HNSW overhead (disabled for original vectors)
   - Returns final top-k results (e.g., `limit = 10`)

**Query Pattern**:
```python
response = client.query_points(
    collection_name="RAG_ColQwen2",
    query=query_embedding,  # From ColQwen2 model
    prefetch=[
        models.Prefetch(query=query_embedding, limit=100, using="mean_pooling_columns"),
        models.Prefetch(query=query_embedding, limit=100, using="mean_pooling_rows"),
    ],
    limit=10,
    using="original"  # Rerank with original multivectors
)
```

## Development Workflows

### Windows Development Setup
The project uses VS Code devcontainer for Windows development:
- [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) references root [Dockerfile](Dockerfile)
- Provides uv package manager and Python environment
- Includes GitHub CLI and Python features pre-installed

### Running the Stack
```bash
cd root/docker
docker-compose up --build
```
Services depend on `vector-db` → `ingestion` → `orchestrator`.

### Adding Documents to Knowledge Base
1. Place PDFs in `data/ingestion/unprocessed/`
2. Restart ingestion service to trigger processing
3. Processed PDFs auto-move to `data/ingestion/processed/`

### Dependency Management
Uses **uv** (Astral SH's package manager) for all containers:
- `pyproject.toml` defines dependencies (no lockfile committed)
- Build: `uv sync` in Dockerfile
- Run: `uv run main.py`

## Project Conventions

### Container Structure Pattern
Every container follows identical structure:
```
container-name/
├── Dockerfile       # uv-based Python 3.14 slim
├── main.py          # Entry point with main() function
├── pyproject.toml   # uv project definition
└── README.md        # Service-specific documentation
```

### Environment Configuration
- Qdrant URL injected via `QDRANT_URL` environment variable
- Ingestion expects data volume at `/app/data`
- Base Dockerfile ([Dockerfile](Dockerfile)) handles Zscaler certificates for corporate proxies

### Python Requirements
- Python 3.13+ (containers use 3.14-slim-trixie)
- Key dependencies: LangGraph, ColPali-Engine, Qdrant-Client, PyTorch

## Integration Points

### Qdrant Vector Store
- Collection: `RAG_ColQwen2` with multivector configuration
- Connection: HTTP via docker-compose service name (`vector-db:6333`)
- Volume: `data/vector-db/` persists embeddings

### Cross-Service Communication
- Services communicate via Docker network
- LangGraph will orchestrate agent interactions (orchestrator selects appropriate agent based on query)
- UI will communicate with orchestrator (protocol TBD, not yet implemented)
- Future API key requirements: Jama, SharePoint, Qdrant Cloud integrations

## Current State & TODOs

**Implemented**:
- Full ingestion pipeline with ColQwen2 embeddings
- Qdrant vector database setup
- Docker infrastructure

**Stubs (only print statements)**:
- Orchestrator LangGraph workflows (general-purpose agent selector)
- Knowledge agent RAG retrieval (implement two-stage retrieval pattern above)
- UI frontend

**MVP Goal**: Test RAG retrieval with ECSS standard documents using the two-stage retrieval pattern.

When extending stubs, maintain strict layer separation and use dependency injection for service URLs.
