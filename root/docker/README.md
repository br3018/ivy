# Ivy AI Orchestration System - Deployment Guide

This guide covers deploying the Ivy system with Open WebUI, Ollama, and the orchestrator service.

## Quick Start

### 1. Initial Setup

```bash
cd root/docker

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

### 2. Configure Environment

Edit `.env` to set your LLM provider:

**For local Ollama (default):**
```bash
LLM_PROVIDER=local
OLLAMA_MODEL=llama3.1
```

**For Claude (Anthropic):**
```bash
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### 3. Start Services

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 4. Install Ollama Model (if using local LLM)

```bash
# Wait for services to start, then pull a model
docker exec ollama ollama pull llama3.1

# Verify installation
docker exec ollama ollama list
```

### 5. Access Open WebUI

Navigate to: **http://localhost:3000**

1. Create an account (first user becomes admin)
2. Select "ivy-orchestrator" from model dropdown
3. Start chatting!

## Service Architecture

```
Open WebUI (port 3000)
    ↓
Orchestrator (port 8000) ← OpenAI-compatible API
    ↓
    ├─→ Ollama (port 11434) ← Local LLM
    ├─→ Claude API ← Anthropic
    └─→ Knowledge Agent (port 8001)
            ↓
        Qdrant (port 6333) ← Vector database
```

## Service Details

| Service | Port | Description | Volume |
|---------|------|-------------|--------|
| **open-webui** | 3000 | Chat interface | `data/open-webui/` |
| **orchestrator** | 8000 | API coordinator | - |
| **ollama** | 11434 | Local LLM service | `data/ollama/` |
| **knowledge** | 8001 | RAG agent | `data/knowledge-cache/` |
| **vector-db** | 6333, 6334 | Qdrant vector DB | `data/vector-db/` |
| **ingestion** | - | PDF processor | `data/ingestion/` |

## GPU Configuration

### Ollama with NVIDIA GPU

For GPU acceleration with Ollama, edit `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:latest
  ports:
    - "11434:11434"
  volumes:
    - ../../data/ollama:/root/.ollama
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

**Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker configured for NVIDIA runtime

**Install NVIDIA Container Toolkit (Ubuntu/Debian):**
```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Windows WSL2 with GPU:**
1. Install NVIDIA GPU drivers for WSL2
2. Install NVIDIA Container Toolkit in WSL2
3. Uncomment GPU configuration in docker-compose.yml

See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

### Knowledge Agent with GPU

The knowledge agent automatically uses GPU if available (detected via PyTorch). No additional configuration needed.

## Model Switching

### Switch Between Local and Claude

1. Stop the orchestrator:
   ```bash
   docker-compose stop orchestrator
   ```

2. Edit `.env`:
   ```bash
   # Switch to Claude
   LLM_PROVIDER=claude
   ANTHROPIC_API_KEY=sk-ant-...
   
   # Or switch to local
   LLM_PROVIDER=local
   OLLAMA_MODEL=mistral
   ```

3. Restart orchestrator:
   ```bash
   docker-compose start orchestrator
   ```

### Change Ollama Model

1. Pull new model:
   ```bash
   docker exec ollama ollama pull mistral
   ```

2. Update `.env`:
   ```bash
   OLLAMA_MODEL=mistral
   ```

3. Restart orchestrator:
   ```bash
   docker-compose restart orchestrator
   ```

## Adding Documents to Knowledge Base

1. Place PDF files in `data/ingestion/unprocessed/`

2. Restart ingestion service to trigger processing:
   ```bash
   docker-compose restart ingestion
   ```

3. Monitor processing:
   ```bash
   docker-compose logs -f ingestion
   ```

4. Processed PDFs automatically move to `data/ingestion/processed/`

## Troubleshooting

### Ollama model not responding

```bash
# Check if model is installed
docker exec ollama ollama list

# Pull model if missing
docker exec ollama ollama pull llama3.1

# Test Ollama directly
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1",
  "prompt": "Hello!",
  "stream": false
}'
```

### Knowledge agent not returning results

```bash
# Check Qdrant collection
curl http://localhost:6333/collections/RAG_ColQwen2

# Verify documents exist
curl http://localhost:6333/collections/RAG_ColQwen2/points/count

# Test knowledge agent directly
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "payload fairing", "limit": 5}'
```

### Open WebUI can't connect to orchestrator

```bash
# Check orchestrator health
curl http://localhost:8000/health

# View orchestrator logs
docker-compose logs orchestrator

# Verify Open WebUI configuration
docker-compose exec open-webui env | grep OPENAI
```

### Claude API errors

```bash
# Verify API key is set
docker-compose exec orchestrator env | grep ANTHROPIC

# Test Claude API directly
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v

# Stop specific service
docker-compose stop orchestrator
```

## Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f orchestrator
docker-compose logs -f knowledge
docker-compose logs -f ollama

# Last 100 lines
docker-compose logs --tail=100 orchestrator
```

## Remote GPU Server Deployment

To run the orchestrator on a remote GPU server and connect from your local machine:

### Option 1: Expose Orchestrator Port

**On GPU server:**
1. Configure firewall to allow port 8000
2. Update `docker-compose.yml` to bind to all interfaces (already configured: `0.0.0.0:8000`)
3. Start services

**On local machine:**
1. Run only Open WebUI locally
2. Set environment variable:
   ```bash
   OPENAI_API_BASE_URL=http://gpu-server-ip:8000/v1
   ```

### Option 2: SSH Tunnel

**On local machine:**
```bash
# Forward remote port 8000 to local port 8000
ssh -L 8000:localhost:8000 user@gpu-server

# Then run Open WebUI locally with:
OPENAI_API_BASE_URL=http://localhost:8000/v1
```

### Option 3: Reverse Proxy with SSL

**On GPU server, use Caddy:**
```bash
# Install Caddy
sudo apt install -y caddy

# Configure /etc/caddy/Caddyfile
ai.your-domain.com {
    reverse_proxy localhost:8000
}

# Restart Caddy
sudo systemctl restart caddy
```

**On local machine:**
```bash
OPENAI_API_BASE_URL=https://ai.your-domain.com/v1
```

## Data Persistence

All data is stored in the `data/` directory:

```
data/
├── ingestion/              # PDF processing
│   ├── unprocessed/       # Drop PDFs here
│   ├── processed/         # Completed PDFs
│   └── huggingface-cache/ # ColQwen2 models (ingestion)
├── knowledge-cache/        # ColQwen2 models (knowledge agent)
├── ollama/                # Ollama models
├── open-webui/            # User data and chat history
└── vector-db/             # Qdrant embeddings
```

**Backup important data:**
```bash
# Backup vector database and user data
tar -czf ivy-backup-$(date +%Y%m%d).tar.gz data/vector-db data/open-webui
```

## Performance Tuning

### Ollama Settings

Configure Ollama environment variables in docker-compose.yml:

```yaml
ollama:
  environment:
    - OLLAMA_NUM_PARALLEL=2        # Concurrent requests
    - OLLAMA_MAX_LOADED_MODELS=1   # Memory management
    - OLLAMA_FLASH_ATTENTION=1     # Enable flash attention (faster)
```

### Knowledge Agent Batch Size

Edit `containers/agents/knowledge/main.py` to adjust query limits:

```python
# Increase prefetch for better recall (slower)
prefetch=[
    models.Prefetch(query=query_embedding, limit=200, using="mean_pooling_columns"),
    models.Prefetch(query=query_embedding, limit=200, using="mean_pooling_rows"),
]
```

## Security Considerations

- **Never commit `.env` file** - contains API keys
- **Restrict port access** - only expose necessary ports
- **Use HTTPS in production** - SSL termination with reverse proxy
- **Enable Open WebUI authentication** - configure in UI settings
- **Rotate API keys regularly** - update `.env` and restart

## Next Steps

1. Add more documents to the knowledge base
2. Experiment with different Ollama models
3. Configure Open WebUI settings and personas
4. Implement LangGraph workflows in orchestrator
5. Add custom tools and function calling

For more details, see individual service READMEs:
- [Orchestrator README](containers/orchestrator/README.md)
- [Ingestion README](containers/ingestion/README.md)
- [Knowledge Agent README](containers/agents/knowledge/README.md)