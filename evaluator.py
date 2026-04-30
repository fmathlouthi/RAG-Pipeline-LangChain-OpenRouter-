"""
evaluator.py
Single Responsibility: Given a question, the generated answer, and the
expected answer, ask the LLM to judge correctness and return a structured
EvaluationResult.
"""

import os
import json
from dataclasses import dataclass

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

EVAL_PROMPT_TEMPLATE = """
You are a strict but fair evaluator for a question-answering system.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Evaluate whether the Generated Answer is correct based on the Expected Answer.
Respond ONLY with valid JSON in this exact format, no extra text:
{{
  "correct": true or false,
  "score": a float between 0.0 and 1.0,
  "reasoning": "one sentence explaining your judgment"
}}
""".strip()


@dataclass
class EvaluationResult:
    question: str
    expected_answer: str
    generated_answer: str
    correct: bool
    score: float          # 0.0 – 1.0
    reasoning: str


def _get_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")


def _build_eval_prompt(question: str, expected_answer: str, generated_answer: str) -> str:
    return EVAL_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )


def _build_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,   # deterministic for evaluation
    }


def _build_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _extract_json_from_llm_output(raw_output: str) -> dict:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # Fallback: if the model wrapped JSON in markdown fences
        import re

        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"correct": False, "score": 0.0, "reasoning": raw_output}


def _parse_evaluation_result(
    question: str,
    expected_answer: str,
    generated_answer: str,
    parsed: dict,
) -> EvaluationResult:
    return EvaluationResult(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        correct=bool(parsed.get("correct", False)),
        score=float(parsed.get("score", 0.0)),
        reasoning=parsed.get("reasoning", ""),
    )


def evaluate_answer(
    question: str,
    expected_answer: str,
    generated_answer: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> EvaluationResult:
    """
    Use the LLM to judge whether `generated_answer` matches `expected_answer`.
    Returns a structured EvaluationResult.
    """
    import httpx

    key = _get_api_key(api_key)
    prompt = _build_eval_prompt(question, expected_answer, generated_answer)
    payload = _build_payload(model, prompt)
    headers = _build_headers(key)

    response = httpx.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()

    raw_output = response.json()["choices"][0]["message"]["content"].strip()
    parsed = _extract_json_from_llm_output(raw_output)
    return _parse_evaluation_result(question, expected_answer, generated_answer, parsed)


def evaluate_batch(
    qa_pairs: list[dict],   # list of {"question": ..., "expected_answer": ...}
    pipeline_fn,            # callable(document_path, question) -> str
    document_path: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[EvaluationResult]:
    """
    Run the full pipeline + evaluator over a list of QA pairs.

    qa_pairs format:
        [
          {"question": "What is X?", "expected_answer": "X is ..."},
          ...
        ]
    """
    results = []
    for item in qa_pairs:
        question = item["question"]
        expected = item["expected_answer"]
        generated = pipeline_fn(document_path, question)
        result = evaluate_answer(question, expected, generated, model=model, api_key=api_key)
        results.append(result)
    return results


def print_eval_report(results: list[EvaluationResult]) -> None:
    """Pretty-print a summary of evaluation results."""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    avg_score = sum(r.score for r in results) / total if total else 0.0

    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT  ({correct}/{total} correct, avg score: {avg_score:.2f})")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        status = "✓ PASS" if r.correct else "✗ FAIL"
        print(f"\n[{i}] {status}  (score: {r.score:.2f})")
        print(f"  Q        : {r.question}")
        print(f"  Expected : {r.expected_answer}")
        print(f"  Got      : {r.generated_answer}")
        print(f"  Reason   : {r.reasoning}")

    print("\n" + "=" * 60)
