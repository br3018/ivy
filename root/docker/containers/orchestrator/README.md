# Orchestrator Service

The Orchestrator is the central coordinator for the Ivy AI system, providing an OpenAI-compatible API that routes requests to either local LLM (Ollama) or Claude (Anthropic) and integrates with the knowledge agent for RAG capabilities.

## Features

- **OpenAI-compatible API**: `/v1/chat/completions` and `/v1/models` endpoints
- **Streaming responses**: Real-time token streaming using Server-Sent Events (SSE)
- **Flexible LLM providers**: Switch between local Ollama models and Claude
- **RAG integration**: Automatically queries knowledge agent for relevant documents
- **Environment-based configuration**: Easy model switching via `.env` file

## API Endpoints

### POST `/v1/chat/completions`

OpenAI-compatible chat completions endpoint with streaming support.

**Request body:**
```json
{
  "model": "ivy-orchestrator",
  "messages": [
    {"role": "user", "content": "What is the payload fairing design?"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true
}
```

**Response:** Server-Sent Events (SSE) stream in OpenAI format

### GET `/v1/models`

Lists available models for Open WebUI dropdown.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "ivy-orchestrator",
      "object": "model",
      "created": 1704150000,
      "owned_by": "ivy"
    }
  ]
}
```

### GET `/health`

Health check endpoint returning current configuration.

**Response:**
```json
{
  "status": "healthy",
  "llm_provider": "local",
  "model": "llama3.1",
  "knowledge_agent_url": "http://knowledge:8001"
}
```

## Configuration

### Environment Variables

All configuration is managed through environment variables in the `.env` file:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_PROVIDER` | LLM provider (`local` or `claude`) | `local` | Yes |
| `OLLAMA_BASE_URL` | Ollama service URL | `http://ollama:11434` | If using local |
| `OLLAMA_MODEL` | Ollama model name | `llama3.1` | If using local |
| `ANTHROPIC_API_KEY` | Claude API key | - | If using claude |
| `KNOWLEDGE_AGENT_URL` | Knowledge agent URL | `http://knowledge:8001` | Yes |

### Switching Between Models

#### Using Local Ollama Model

1. Edit `root/docker/.env`:
   ```bash
   LLM_PROVIDER=local
   OLLAMA_MODEL=llama3.1
   ```

2. Restart orchestrator:
   ```bash
   docker-compose restart orchestrator
   ```

#### Using Claude (Anthropic)

1. Edit `root/docker/.env`:
   ```bash
   LLM_PROVIDER=claude
   ANTHROPIC_API_KEY=sk-ant-api03-...
   ```

2. Restart orchestrator:
   ```bash
   docker-compose restart orchestrator
   ```

## Ollama Model Management

### Installing Models

After starting the stack, install Ollama models:

```bash
# List available models
docker exec ollama ollama list

# Pull a new model
docker exec ollama ollama pull llama3.1
docker exec ollama ollama pull mistral
docker exec ollama ollama pull codellama

# Remove a model
docker exec ollama ollama rm llama3.1
```

### Available Models

Popular models for Ollama:
- `llama3.1` - Meta's Llama 3.1 (8B, 70B, 405B)
- `llama3.2` - Meta's Llama 3.2 (1B, 3B)
- `mistral` - Mistral 7B
- `codellama` - Code-specialized Llama
- `qwen2.5` - Alibaba's Qwen 2.5

See full list at: https://ollama.com/library

## Architecture

### Request Flow

1. **Open WebUI** → sends chat request to `/v1/chat/completions`
2. **Orchestrator** → queries knowledge agent if query needs context
3. **Knowledge Agent** → returns relevant documents from Qdrant
4. **Orchestrator** → augments messages with retrieved context
5. **Orchestrator** → routes to Ollama or Claude based on `LLM_PROVIDER`
6. **LLM** → streams response tokens back through orchestrator
7. **Open WebUI** → displays streaming response in real-time

### RAG Integration

The orchestrator automatically queries the knowledge agent for user messages. Retrieved documents are injected as system context:

```
System: Relevant documents from knowledge base:

[Document 1 - ECSS-E-ST-32-08C Page 45]:
(Score: 0.8542)

[Document 2 - ECSS-E-ST-32-08C Page 46]:
(Score: 0.8234)

Use this context to help answer the user's question.

User: What is the payload fairing design?
```

## Dependencies

- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **anthropic**: Claude API client
- **httpx**: Async HTTP client for Ollama
- **sse-starlette**: Server-Sent Events support
- **langgraph**: AI orchestration (planned integration)
- **python-dotenv**: Environment variable management

## Development

### Running Locally

```bash
cd root/docker/containers/orchestrator
uv sync
uv run main.py
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ivy-orchestrator",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Future Enhancements

- **LangGraph integration**: Sophisticated agent routing and multi-step reasoning
- **Multiple model exposure**: Expose different models for UI selection
- **Hot-reload configuration**: Change models without restart
- **Authentication**: API key-based access control
- **Usage tracking**: Token counting and rate limiting
- **Tool use**: Function calling for specialized tasks