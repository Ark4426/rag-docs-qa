"""FastAPI layer. Run: uvicorn api:app --reload   (Swagger at /docs)"""
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag import RAG, SUPPORTED, settings

app = FastAPI(title="RAG Documents Q&A")
rag = RAG()


class QueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(4, ge=1, le=20)


@app.exception_handler(ConnectionError)
@app.exception_handler(httpx.ConnectError)
async def model_unreachable(request: Request, exc: Exception):
    return JSONResponse(status_code=503, content={"detail": f"Model backend unreachable at {settings.ollama_url}: {exc}"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return rag.stats()


@app.post("/ingest")
def ingest(files: list[UploadFile]):
    names = [Path(f.filename or "").name for f in files]  # .name strips any path -> no traversal out of data_dir
    bad = [n or "<unnamed>" for n in names if Path(n).suffix.lower() not in SUPPORTED]
    if bad:
        raise HTTPException(415, f"Unsupported files {bad}. Supported: {sorted(SUPPORTED)}")
    limit = settings.max_upload_mb * 1024 * 1024
    incoming = settings.data_dir / ".incoming"  # staged here; only successfully ingested files land in data_dir
    incoming.mkdir(parents=True, exist_ok=True)
    results = []
    for f, name in zip(files, names):
        data = f.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{name} exceeds {settings.max_upload_mb} MB")
        staged = incoming / name
        staged.write_bytes(data)
        try:
            results.append({"file": name, "chunks": rag.ingest(staged)})
            staged.replace(settings.data_dir / name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        finally:
            staged.unlink(missing_ok=True)  # no-op after a successful replace
    return {"ingested": results, **rag.stats()}


@app.post("/query")
def query(q: QueryIn):
    return rag.answer(q.query, q.k)
