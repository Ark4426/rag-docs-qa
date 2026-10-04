# RAG Documents Q&A Service

Implements [`PLAN.md`](PLAN.md). LangChain pipeline: PyMuPDF/python-docx loaders → RecursiveCharacterTextSplitter → Ollama `nomic-embed-text` → FAISS (cosine) → grounded ChatOllama answer with `[n]` citations.

| File | What |
|---|---|
| `rag.py` | settings, loaders (PDF/DOCX/TXT/MD), `RAG` class: ingest (upsert by filename), retrieve, answer, stats |
| `api.py` | FastAPI: `POST /ingest` (multipart), `POST /query`, `GET /health`, `GET /stats`; Swagger at `/docs` |
| `test_rag.py` | offline pytest suite (fake embeddings + fake LLM) |
| `eval/run_eval.py` | Hit-Rate@k + Answer Relevance (LLM-as-judge) over `eval/eval_set.json` |

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
ollama pull nomic-embed-text; ollama pull qwen2.5-coder:3b
.\.venv\Scripts\uvicorn api:app --reload

curl -F "files=@report.pdf" localhost:8000/ingest
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What is the budget?","k":4}'

.\.venv\Scripts\pytest -q
.\.venv\Scripts\python eval\run_eval.py --k 3
```

Docker (Ollama stays on the host):
```
docker build -t rag-qa .
docker run -p 8000:8000 -v %cd%/index:/app/index -v %cd%/data:/app/data rag-qa
```
