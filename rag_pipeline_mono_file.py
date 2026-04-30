"""Simple LangChain RAG pipeline."""

import sys
import json
import os
from functools import lru_cache
from pathlib import Path

from evaluator import evaluate_batch, print_eval_report

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class Loader:
    def run(self, document_path: str):
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
        except ImportError as exc:
            raise ImportError(
                "Install: pip install langchain langchain-community"
            ) from exc

        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {document_path}")
        if path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(path)).load()
        if path.suffix.lower() == ".txt":
            return TextLoader(str(path), encoding="utf-8").load()
        raise ValueError("Only .pdf and .txt are supported.")


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self, docs):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)


class RetrieverBuilder:
    def __init__(self, embedding_model: str, top_k: int):
        self.embedding_model = embedding_model
        self.top_k = top_k

    def run(self, docs):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        store = FAISS.from_documents(docs, embeddings)
        return store.as_retriever(search_kwargs={"k": self.top_k})


class ContextBuilder:
    @staticmethod
    def run(retriever, question: str) -> str:
        docs = retriever.invoke(question)
        return "\n\n---\n\n".join(d.page_content for d in docs)


class Generator:
    def __init__(self, llm_model: str, base_url: str):
        self.llm_model = llm_model
        self.base_url = base_url

    def _llm(self):
        from langchain_openai import ChatOpenAI

        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise EnvironmentError("OPENROUTER_API_KEY is missing.")
        return ChatOpenAI(
            model=self.llm_model,
            api_key=key,
            base_url=self.base_url,
            temperature=0.2,
        )

    def run(self, question: str, context: str) -> str:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            "Use only this context. If missing, say 'I don't know'.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        chain = prompt | self._llm() | StrOutputParser()
        return chain.invoke({"context": context, "question": question}).strip()


class RAGPipeline:
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.loader = Loader()
        self.chunker = Chunker(CHUNK_SIZE, CHUNK_OVERLAP)
        self.retriever_builder = RetrieverBuilder(EMBEDDING_MODEL, TOP_K)
        self.context_builder = ContextBuilder()
        self.generator = Generator(LLM_MODEL, OPENROUTER_BASE_URL)
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        docs = self.loader.run(self.document_path)
        chunks = self.chunker.run(docs)
        print(f"Chunks: {len(chunks)}")
        return self.retriever_builder.run(chunks)

    def answer(self, question: str) -> str:
        context = self.context_builder.run(self.retriever, question)
        return self.generator.run(question, context)


def _print_usage() -> None:
    print(
        "Usage:\n"
        '  python rag_pipeline.py query "<document>" "<question>"\n'
        '  python rag_pipeline.py evaluate "<document>" "<eval_questions.json>"\n'
    )


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@lru_cache(maxsize=4)
def _get_pipeline(document_path: str) -> RAGPipeline:
    return RAGPipeline(document_path)


def run_pipeline(document_path: str, question: str) -> str:
    return _get_pipeline(document_path).answer(question)


def _cmd_query(document_path: str, question: str) -> None:
    print(f"Document: {document_path}")
    print(f"Question: {question}")
    print("-" * 60)
    print(run_pipeline(document_path, question))
    print("-" * 60)


def _cmd_evaluate(document_path: str, eval_json_path: str) -> None:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
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

    cmd = sys.argv[1].lower()
    if cmd == "query" and len(sys.argv) >= 4:
        _cmd_query(sys.argv[2], sys.argv[3])
        return
    if cmd == "evaluate" and len(sys.argv) >= 4:
        _cmd_evaluate(sys.argv[2], sys.argv[3])
        return

    _print_usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
