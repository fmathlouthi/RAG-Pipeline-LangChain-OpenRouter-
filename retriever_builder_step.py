"""Step 3: build a retriever from chunked documents."""


class RetrieverBuilderStep:
    """Create a FAISS retriever with HuggingFace embeddings."""

    def __init__(self, embedding_model: str, top_k: int):
        self.embedding_model = embedding_model
        self.top_k = top_k

    def build(self, split_docs):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "LangChain vector dependencies are missing. Install with: "
                "pip install langchain-community langchain-huggingface"
            ) from exc

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": self.top_k})
