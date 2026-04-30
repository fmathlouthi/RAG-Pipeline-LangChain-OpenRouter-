"""Step 1: load input files into LangChain documents."""

from pathlib import Path


class DocumentLoaderStep:
    """Load .pdf or .txt files."""

    def load(self, document_path: str):
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
        except ImportError as exc:
            raise ImportError(
                "LangChain dependencies are missing. Install with: "
                "pip install langchain langchain-community langchain-openai "
                "langchain-huggingface langchain-text-splitters"
            ) from exc

        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {document_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(path)).load()
        if suffix == ".txt":
            return TextLoader(str(path), encoding="utf-8").load()

        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
