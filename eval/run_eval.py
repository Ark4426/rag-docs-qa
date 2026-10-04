"""Evaluation: Retrieval Hit-Rate@k + Answer Relevance (LLM-as-judge, 1-5).
Builds a throwaway index from eval/docs, so the service's real index is untouched.

    python eval/run_eval.py [--k 4]
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from rag import RAG  # noqa: E402

JUDGE = """Rate how well the ANSWER answers the QUESTION, compared to the REFERENCE.
5 = fully correct and on point, 3 = partially correct, 1 = wrong, irrelevant or "I don't know".
QUESTION: {q}
REFERENCE: {ref}
ANSWER: {ans}
Reply with a single digit 1-5 only."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    k = ap.parse_args().k
    cases = json.loads((HERE / "eval_set.json").read_text("utf-8"))
    with tempfile.TemporaryDirectory() as d:
        rag = RAG(Path(d))
        for doc in sorted((HERE / "docs").iterdir()):
            rag.ingest(doc)
        hits, scores = 0, []
        for c in cases:
            out = rag.answer(c["question"], k)
            hit = c["source"] in {s["file"] for s in out["sources"]}
            verdict = rag.llm.invoke(JUDGE.format(q=c["question"], ref=c["expected"], ans=out["answer"])).content
            m = re.search(r"[1-5]", verdict)
            score = int(m.group()) if m else 1  # unparseable judge reply counts as worst
            hits += hit
            scores.append(score)
            print(f"{'HIT ' if hit else 'MISS'} rel={score} {out['latency_ms']:>6}ms | {c['question']}\n"
                  f"      -> {out['answer'][:140]!r}")
    print(f"\nHit-Rate@{k}: {hits / len(cases):.2%}   "
          f"Answer Relevance: {sum(scores) / len(scores):.2f}/5 ({(sum(scores) / len(scores) - 1) / 4:.0%})")


if __name__ == "__main__":
    main()
