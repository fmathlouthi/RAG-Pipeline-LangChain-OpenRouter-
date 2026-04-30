# RAG Pipeline (Previous Structure: One Class Per File)

This document describes the previous project layout where each pipeline step
was implemented in a separate file/class.

## Architecture Overview

The RAG pipeline was split into dedicated classes, each with one clear job:

1. Load document
2. Chunk document
3. Build retriever
4. Build context
5. Generate answer
6. Orchestrate all steps

This design is useful when you want strict separation of concerns and easier
unit testing per step.

## Previous File Layout

- `rag_pipeline.py` - CLI entrypoint and pipeline cache
- `rag_orchestrator.py` - `RAGPipeline` orchestrator class
- `document_loader_step.py` - `DocumentLoaderStep`
- `document_chunker_step.py` - `DocumentChunkerStep`
- `retriever_builder_step.py` - `RetrieverBuilderStep`
- `context_retriever_step.py` - `ContextRetrieverStep`
- `answer_generator_step.py` - `AnswerGeneratorStep`
- `evaluator.py` - evaluation logic/report

## Responsibilities

### `DocumentLoaderStep`

- Input: document path
- Output: LangChain documents
- Supports `.txt` and `.pdf`

### `DocumentChunkerStep`

- Input: loaded documents
- Output: split chunks
- Uses chunk size + overlap settings

### `RetrieverBuilderStep`

- Input: chunks
- Output: retriever
- Uses HuggingFace embeddings + FAISS

### `ContextRetrieverStep`

- Input: retriever + question
- Output: merged context string

### `AnswerGeneratorStep`

- Input: question + context
- Output: final answer text
- Uses OpenRouter via LangChain `ChatOpenAI`

### `RAGPipeline` (Orchestrator)

- Composes all step classes
- Builds retriever once per document
- Exposes `answer(question)` for the CLI

## Typical Flow

```text
document_path
  -> DocumentLoaderStep
  -> DocumentChunkerStep
  -> RetrieverBuilderStep
question
  -> ContextRetrieverStep
  -> AnswerGeneratorStep
  -> final answer
```

## Why This Structure

- Cleaner SRP boundaries
- Easier to test each step in isolation
- Easier to replace one step (e.g., vector store or model) without touching others

## How to Use (when this structure is active)

```powershell
python rag_pipeline.py query "notes.txt" "What is the main idea?"
python rag_pipeline.py evaluate "notes.txt" "eval_questions.json"
```

## Note

Your current codebase has been simplified back to a single-file class layout in
`rag_pipeline.py`. This file documents the previous modular layout for reference.
