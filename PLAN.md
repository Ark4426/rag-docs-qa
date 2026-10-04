# Project 1: RAG Documents Q&A Service – Implementation Plan

## 1. Overview

Build a Retrieval-Augmented Generation (RAG) service that allows users to upload documents and ask natural-language questions. The system will load documents, chunk and embed them, store vectors in a FAISS index, retrieve relevant context, and generate grounded answers using an LLM. It will be exposed via FastAPI, containerized with Docker, covered by pytest unit tests, and include a small LLM evaluation set to measure retrieval hit-rate and answer relevance.

## 2. Goals

- Implement an end-to-end LangChain RAG pipeline (document loading, chunking, embeddings, vector store, retrieval, answer generation).
- Serve the pipeline via a REST API using FastAPI.
- Persist the FAISS vector index and document metadata to disk for reuse across restarts.
- Enforce grounded, source-cited answers (reduce hallucinations).
- Provide pytest-based unit tests for core components.
- Include a small evaluation set with metrics: **Retrieval Hit-Rate @k** and **Answer Relevance** (LLM-as-judge).
- Containerize the service with Docker for reproducible local/dev deployment.

## 3. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core implementation |
| RAG Framework | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) | Orchestration of RAG pipeline |
| Document Loaders | [`pymupdf`](https://pymupdf.readthedocs.io/) (PDF), [`python-docx`](https://python-docx.readthedocs.io/), [`unstructured`](https://github.com/Unstructured-IO/unstructured) (optional), [`markitdown`](https://github.com/microsoft/markitdown) (optional) | Extract clean text from common formats (PDF, TXT, MD, DOCX) |
| Text Splitting | `RecursiveCharacterTextSplitter` (LangChain) | Semantic/overlap-aware chunking |
| Embeddings | [HuggingFace](https://huggingface.co/) via `langchain-huggingface` (e.g. `BAAI/bge-m3`, `nomic-ai/nomic-embed-text-v2`, `sentence-transformers/all-MiniLM-L6-v2`) or [Ollama](https://ollama.com/) embeddings (`nomic-embed-text`) | Convert chunks to dense vectors |
| Vector Store | [FAISS](https://faiss.ai/) (`faiss-cpu`) | Local, fast similarity search (GPU optional: `faiss-gpu`) |
| LLM | [Ollama](https://ollama.com/) (local-first, e.g. `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct`, `mistral-nemo`) or OpenRouter/Gemini (API-based) | Answer generation |
| API | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) (`uvicorn[standard]`) | REST endpoints + Swagger/ReDoc (`/docs`) |
| Config | [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) + `.env` | Environment-driven configuration |
| Testing/Eval | [`pytest`](https://docs.pytest.org/), [`pytest-asyncio`](https://pytest-asyncio.readthedocs.io/), [`httpx`](https://www.python-httpx.org/), [`ragas`](https://docs.ragas.io/) (optional) | Unit tests + evaluation |
| Packaging | `pyproject.toml` or `requirements.txt` | Dependency management |
| Containerization | [Docker](https://www.docker.com/), `docker-compose.yml` (optional) | Isolated, reproducible runtime |

## 4. Architecture

The system is split into modular layers to keep ingestion, retrieval, generation, and API concerns decoupled.

| Component | Responsibility |
|---|---|
| **Ingestion** | Document discovery/loading, text normalization, chunking, embedding generation, FAISS index creation/upsert and persistence. |
| **Retriever** | Embed user queries, perform similarity search (top-k), optionally apply MMR/diversity, return ranked chunk documents with scores/metadata. |
| **Generator (RAG Chain)** | Format retrieved context into a strict grounded prompt, call LLM, return answer + source metadata (file, page, chunk_id, text snippet). |
| **API Layer** | Expose ingestion/query/health/stats endpoints, input validation (Pydantic), error handling, request/response schemas. |
| **Persistence** | Store FAISS index (`index.faiss`) + `metadata.json` (doc/chunk mapping, ids) under `./index/`. Keep raw uploads under `./data/` (gitignored). |
| **Config/Core** | Centralized settings, logging, constants, and schema definitions. |

## 5. Data Flow

1. **Upload** – Client uploads one or more documents via `POST /ingest` (multipart).
2. **Load** – Appropriate loader extracts raw text per document (preserve page numbers where available).
3. **Chunk** – `RecursiveCharacterTextSplitter` splits text into overlapping chunks with stable boundaries.
4. **Embed** – Embedding model converts each chunk to a vector (normalize if using cosine/IP).
5. **Index** – Chunks + metadata written to FAISS index; index + `metadata.json` persisted to `./index/` (idempotent/upsert-friendly).
6. **Query** – User sends `{query, k, ...}` to `POST /query`.
7. **Retrieve** – Query embedded → FAISS similarity search returns top-k relevant chunks (with distance/score + source metadata).
8. **Augment** – Retrieved chunks formatted into context block with clear source attribution.
9. **Generate** – Grounded system prompt + context + question sent to LLM → answer generated with citations/sources.
10. **Respond** – API returns `{answer, sources[], latency_ms}` (include scores/snippets as requested).