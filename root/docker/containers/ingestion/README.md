# Ingestion Service

## Overview

The ingestion service processes PDF documents and uploads them to the Qdrant vector database using ColQwen2 multivector embeddings. This service is a critical component of the Ivy AI orchestration system's knowledge layer.

## Architecture

### Embedding Strategy

The service implements a **three-vector approach** for optimal retrieval performance:

1. **Original Embeddings** (~700 vectors/page, 128-dim)
   - Full multivector representation preserving spatial structure
   - HNSW indexing disabled for faster ingestion
   - Used only in the reranking stage of retrieval

2. **Row-Pooled Embeddings** (128-dim)
   - Mean pooled across rows (horizontal aggregation)
   - HNSW indexing enabled for fast similarity search
   - Used in the prefetch stage of retrieval

3. **Column-Pooled Embeddings** (128-dim)
   - Mean pooled across columns (vertical aggregation)
   - HNSW indexing enabled for fast similarity search
   - Used in the prefetch stage of retrieval

### Data Flow

```
1. Monitor: /app/data/unprocessed/ (input PDFs)
2. Convert: PDF → Images (150 DPI)
3. Embed: Images → ColQwen2 embeddings (3 variants)
4. Upload: Embeddings → Qdrant collection "RAG_ColQwen2"
5. Archive: Processed PDFs → /app/data/processed/
```

## Configuration

### Environment Variables

- `QDRANT_URL`: Connection URL for Qdrant instance (e.g., `http://vector-db:6333`)

### Volume Mounts

- `/app/data/unprocessed/`: Input directory for PDFs to process
- `/app/data/processed/`: Archive directory for completed PDFs
- `/app/data/huggingface-cache/`: Model cache directory

### Resource Requirements

- **CPU Mode**: Default configuration (device_map="cpu")
- **GPU Mode**: Modify `device_map="cuda:0"` in main.py for GPU acceleration
- **Memory**: ~4GB RAM recommended per batch

## Usage

### Running the Service

```bash
# Via docker-compose (recommended)
cd root/docker
docker-compose up ingestion

# Standalone container
docker run -v ./data:/app/data -e QDRANT_URL=http://vector-db:6333 ingestion
```

### Adding Documents

1. Place PDF files in `data/unprocessed/`
2. Service automatically processes all PDFs
3. Processed PDFs are moved to `data/processed/`
4. Monitor logs for progress and errors

### Monitoring Progress

The service provides detailed logging:
- Connection status to Qdrant
- Collection creation/verification
- Per-file processing with page counts
- Per-page embedding progress (via tqdm)
- Error messages for failed operations

## Technical Details

### Model Information

- **Model**: ColQwen2 (vidore/colqwen2-v0.1)
- **Precision**: bfloat16
- **Embedding Dimension**: 128
- **Patch-based encoding**: Dynamic grid calculation

### Qdrant Collection Schema

```python
{
  "collection_name": "RAG_ColQwen2",
  "vectors": {
    "original": {
      "size": 128,
      "distance": "COSINE",
      "multivector": "MAX_SIM",
      "hnsw": {"m": 0}  # Disabled
    },
    "mean_pooling_rows": {
      "size": 128,
      "distance": "COSINE",
      "multivector": "MAX_SIM"
      # HNSW enabled (default)
    },
    "mean_pooling_columns": {
      "size": 128,
      "distance": "COSINE",
      "multivector": "MAX_SIM"
      # HNSW enabled (default)
    }
  }
}
```

## References

- **Tutorial**: [Qdrant PDF Retrieval at Scale](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/)
- **Colab Example**: [ColPali/ColQwen2 Tutorial](https://colab.research.google.com/github/qdrant/examples/blob/master/pdf-retrieval-at-scale/ColPali_ColQwen2_Tutorial.ipynb)
- **ColQwen2 Model**: [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1)

## Troubleshooting

### Common Issues

**No PDFs processed**: Ensure PDFs are in `/app/data/unprocessed/` and service has read/write permissions

**Memory errors**: Reduce batch size (PAGES_PER_BATCH constant) or use GPU acceleration

**Qdrant connection failed**: Verify QDRANT_URL environment variable and network connectivity

**Model download issues**: Check Hugging Face cache directory permissions and network access