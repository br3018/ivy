"""
Orchestrator Service for Ivy AI Orchestration System

This service provides an OpenAI-compatible API that routes requests to either
a local Ollama LLM or Claude (Anthropic), integrating with the knowledge agent
for RAG (Retrieval Augmented Generation) capabilities.

Architecture:
- FastAPI server with streaming SSE responses
- LLM provider routing (Ollama or Claude) via environment variables
- Integration with knowledge agent for document retrieval
- OpenAI-compatible /v1/chat/completions and /v1/models endpoints
"""

import os
import json
import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional
from datetime import datetime

import httpx
from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # "local" or "claude"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
KNOWLEDGE_AGENT_URL = os.getenv("KNOWLEDGE_AGENT_URL", "http://knowledge:8001")
MODEL_NAME = "ivy-orchestrator"

# Initialize FastAPI app
app = FastAPI(title="Ivy Orchestrator", version="0.1.0")

# Initialize Anthropic client (if using Claude)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# Pydantic models for OpenAI-compatible API
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True


class QueryRequest(BaseModel):
    query: str
    limit: int = 5


async def query_knowledge_agent(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Query the knowledge agent for relevant documents."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{KNOWLEDGE_AGENT_URL}/query",
                json={"query": query, "limit": limit}
            )
            response.raise_for_status()
            return response.json().get("documents", [])
    except Exception as e:
        print(f"Error querying knowledge agent: {e}")
        return []


async def stream_ollama_response(
    messages: List[Message],
    temperature: float,
    max_tokens: int
) -> AsyncIterator[str]:
    """Stream response from Ollama API."""
    
    # Convert messages to Ollama format
    ollama_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": ollama_messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                if content:
                                    # Format as SSE event in OpenAI format
                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': content}, 'index': 0, 'finish_reason': None}]})}\n\n"
                            
                            if chunk.get("done", False):
                                yield f"data: {json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
                                yield "data: [DONE]\n\n"
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            error_message = f"Error streaming from Ollama: {str(e)}"
            print(error_message)
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            yield "data: [DONE]\n\n"


async def stream_claude_response(
    messages: List[Message],
    temperature: float,
    max_tokens: int
) -> AsyncIterator[str]:
    """Stream response from Claude API."""
    
    if not anthropic_client:
        yield f"data: {json.dumps({'error': 'Claude API key not configured'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    try:
        # Convert messages - separate system message if present
        system_message = None
        claude_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append({"role": msg.role, "content": msg.content})
        
        # Stream from Claude
        async with anthropic_client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=claude_messages,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    # Format as SSE event in OpenAI format
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': text}, 'index': 0, 'finish_reason': None}]})}\n\n"
            
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        error_message = f"Error streaming from Claude: {str(e)}"
        print(error_message)
        yield f"data: {json.dumps({'error': error_message})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with streaming support.
    Routes to either Ollama or Claude based on LLM_PROVIDER environment variable.
    """
    
    # Check for RAG query (simple heuristic - can be enhanced with LangGraph)
    user_message = next((msg.content for msg in request.messages if msg.role == "user"), "")
    
    # Query knowledge agent if the message looks like it needs context
    # TODO: Integrate LangGraph for more sophisticated routing
    documents = []
    if user_message and len(user_message) > 10:  # Simple filter
        documents = await query_knowledge_agent(user_message, limit=5)
    
    # Augment messages with retrieved context if available
    messages = list(request.messages)
    if documents:
        context = "\n\n".join([
            f"[Document {i+1} - {doc.get('source', 'Unknown')} Page {doc.get('page_index', '?')}]:\n(Score: {doc.get('score', 0):.4f})"
            for i, doc in enumerate(documents)
        ])
        
        # Insert context before the last user message
        context_message = Message(
            role="system",
            content=f"Relevant documents from knowledge base:\n\n{context}\n\nUse this context to help answer the user's question."
        )
        messages.insert(-1, context_message)
    
    # Route to appropriate LLM provider
    if LLM_PROVIDER.lower() == "claude":
        return StreamingResponse(
            stream_claude_response(messages, request.temperature, request.max_tokens),
            media_type="text/event-stream"
        )
    else:  # default to local Ollama
        return StreamingResponse(
            stream_ollama_response(messages, request.temperature, request.max_tokens),
            media_type="text/event-stream"
        )


@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint.
    Returns single model for Open WebUI dropdown.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ivy",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None,
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_provider": LLM_PROVIDER,
        "model": OLLAMA_MODEL if LLM_PROVIDER == "local" else "claude-sonnet-4",
        "knowledge_agent_url": KNOWLEDGE_AGENT_URL
    }


def main():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
