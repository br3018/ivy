# ivy
AI orchestration service using LangGraph

## Key Concepts
Project is split into levels defined by strict interfaces

### UI Layer
User interface handles HMI and passes queries and results to the orchestrator layer

### Orchestrator Layer
LLM which handles user queries passed to it by UI layer. Orchestrator delegates tasks to lower level agents and processes

### Agent Layer
LLMs with appropriate tuning and tools to provide accurate responses to orchestrator

### Knowledge Layer
Databases and integrations to knowledge stores accessible by agents. Primary knowledge store is vectorized RAG database of documents

## Structure

### root

Root directory for project

### root/docker

Docker elements of project

### root/docker/containers

Containerized applications for project

#### agents

LLM agents for project, called by orchestrator

#### ingestion

Service for processing data from documents and storing in vector database for later retrival

#### orchestrator

LLM agent responsible for organizing response to user query 

#### ui

Web page for handling user input and passing to orchestrator

### srv

Server directory of project - handles store of data used by project

#### vector-db

Volume for storing vectorized database of documents

#### doc-lib 

Volume for storing documents to be ingested into knowledge store