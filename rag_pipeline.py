"""
rag_pipeline.py
Orchestrator: wires all SRP modules together into a complete RAG pipeline.

Commands:
    query    – ask a single question against a document
    evaluate – run a JSON file of Q&A pairs and score the pipeline

Usage:
    python rag_pipeline.py query   <document_path> "<question>"
    python rag_pipeline.py evaluate <document_path> <eval_json_path>

Examples:
    python rag_pipeline.py query   notes.txt "What embedding model is used?"
    python rag_pipeline.py query   report.pdf "What is the main conclusion?"
    python rag_pipeline.py evaluate notes.txt eval_questions.json
"""

import sys
import json
from functools import lru_cache

from evaluator import evaluate_batch, print_eval_report
from rag_orchestrator import RAGPipeline


# ── tuneable parameters ────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # local, no API key needed
LLM_MODEL = "openai/gpt-4o-mini"  # free on OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# ──────────────────────────────────────────────────────────────────────────────


def _print_usage() -> None:
    print(
        "Usage:\n"
        "  python rag_pipeline.py query    <document> \"<question>\"\n"
        "  python rag_pipeline.py evaluate <document> <eval_questions.json>\n"
    )


def _load_env_file() -> None:
    """Load variables from local .env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@lru_cache(maxsize=4)
def _get_pipeline(document_path: str) -> RAGPipeline:
    # Cache per document path so evaluate mode doesn't rebuild index for each question.
    return RAGPipeline(
        document_path=document_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        openrouter_base_url=OPENROUTER_BASE_URL,
    )


def run_pipeline(document_path: str, question: str) -> str:
    pipeline = _get_pipeline(document_path)
    answer = pipeline.answer(question)
    print("[Done]\n")
    return answer


def _cmd_query(document_path: str, question: str) -> None:
    print("=" * 60)
    print(f"  Document : {document_path}")
    print(f"  Question : {question}")
    print("=" * 60)
    answer = run_pipeline(document_path, question)
    print("─" * 60)
    print("Answer:")
    print(answer)
    print("─" * 60)


def _cmd_evaluate(document_path: str, eval_json_path: str) -> None:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print("=" * 60)
    print(f"  Document    : {document_path}")
    print(f"  Eval file   : {eval_json_path}")
    print(f"  Questions   : {len(qa_pairs)}")
    print("=" * 60)

    results = evaluate_batch(
        qa_pairs=qa_pairs,
        pipeline_fn=run_pipeline,
        document_path=document_path,
        model=LLM_MODEL,
    )
    print_eval_report(results)


def main() -> None:
    _load_env_file()

    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "query":
        if len(sys.argv) < 4:
            print('Usage: python rag_pipeline.py query <document> "<question>"')
            sys.exit(1)
        _cmd_query(document_path=sys.argv[2], question=sys.argv[3])
        return

    if command == "evaluate":
        if len(sys.argv) < 4:
            print("Usage: python rag_pipeline.py evaluate <document> <eval_questions.json>")
            sys.exit(1)
        _cmd_evaluate(document_path=sys.argv[2], eval_json_path=sys.argv[3])
        return

    print(f"Unknown command: '{command}'\n")
    _print_usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
