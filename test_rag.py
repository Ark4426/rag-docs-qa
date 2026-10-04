"""Unit tests, fully offline (fake embeddings + fake LLM). Run: pytest -q"""
from string import ascii_lowercase

import docx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api
from rag import NO_ANSWER, RAG, load


class LetterEmbeddings(Embeddings):
    """Deterministic bag-of-letters vectors: texts sharing letters are similar."""
    def embed_query(self, text):
        return [text.lower().count(c) + 0.01 for c in ascii_lowercase]

    def embed_documents(self, texts):
        return list(map(self.embed_query, texts))


@pytest.fixture
def rag(tmp_path):
    return RAG(tmp_path / "index", LetterEmbeddings(), FakeListChatModel(responses=["Zebras live in zoos [1]."]))


def write(path, text):
    path.write_text(text, "utf-8")
    return path


def test_loaders(tmp_path):
    pdf = tmp_path / "a.pdf"
    d = pymupdf.open()
    for t in ["page one", "page two"]:
        d.new_page().insert_text((72, 72), t)
    d.save(pdf)
    d.close()
    assert [(x.page_content, x.metadata["page"]) for x in load(pdf)] == [("page one", 1), ("page two", 2)]

    dx = docx.Document()
    dx.add_paragraph("hello   docx")
    dx.save(tmp_path / "b.docx")
    assert load(tmp_path / "b.docx")[0].page_content == "hello docx"
    assert load(write(tmp_path / "c.md", "# Title\n\n\n\nbody"))[0].page_content == "# Title\n\nbody"
    with pytest.raises(ValueError):
        load(write(tmp_path / "d.exe", "x"))


def test_ingest_retrieve_upsert_persist(rag, tmp_path):
    rag.ingest(write(tmp_path / "zoo.txt", "zebra zoo zigzag"))
    rag.ingest(write(tmp_path / "food.txt", "apple banana cabbage"))
    doc, score = rag.retrieve("zzz", k=1)[0]
    assert doc.metadata["source"] == "zoo.txt" and -1 <= score <= 1
    rag.ingest(write(tmp_path / "zoo.txt", "zebra zoo zigzag v2"))  # re-upload replaces, no duplicates
    assert rag.stats() == {"documents": 2, "chunks": 2, "files": ["food.txt", "zoo.txt"]}
    reloaded = RAG(rag.dir, LetterEmbeddings(), FakeListChatModel(responses=["x"]))
    assert reloaded.stats() == rag.stats() and reloaded.retrieve("zzz", 1)[0][0].page_content.endswith("v2")


def test_answer(rag, tmp_path):
    assert rag.answer("anything")["answer"] == NO_ANSWER  # empty index never calls the LLM
    rag.ingest(write(tmp_path / "zoo.txt", "zebra zoo zigzag"))
    out = rag.answer("where do zebras live?", k=1)
    assert out["answer"] == "Zebras live in zoos [1]."
    assert out["sources"][0]["file"] == "zoo.txt" and out["sources"][0]["chunk_id"] == "zoo.txt#0"


def test_api(rag, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "rag", rag)
    monkeypatch.setattr(api.settings, "data_dir", tmp_path / "data")
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/ingest", files=[("files", ("../../evil/zoo.txt", b"zebra zoo", "text/plain"))])
    assert r.status_code == 200 and r.json()["files"] == ["zoo.txt"]
    assert (tmp_path / "data" / "zoo.txt").exists()  # path components stripped
    assert c.post("/ingest", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 415
    for bad in ("broken.pdf", "broken.docx", "blank.txt"):
        r = c.post("/ingest", files=[("files", (bad, b"garbage" if "broken" in bad else b"  ", "x/y"))])
        assert r.status_code == 422, (bad, r.text)
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [".incoming", "zoo.txt"]  # rejects not kept
    assert not any((tmp_path / "data" / ".incoming").iterdir())
    assert c.post("/query", json={"query": "zebra", "k": 0}).status_code == 422
    body = c.post("/query", json={"query": "zebra", "k": 1}).json()
    assert set(body) == {"answer", "sources", "latency_ms"} and body["sources"][0]["file"] == "zoo.txt"
