"""RAG orchestrator that only coordinates step classes."""

from answer_generator_step import AnswerGeneratorStep
from context_retriever_step import ContextRetrieverStep
from document_chunker_step import DocumentChunkerStep
from document_loader_step import DocumentLoaderStep
from retriever_builder_step import RetrieverBuilderStep


class RAGPipeline:
    """Orchestrator that only calls step-class methods."""

    def __init__(
        self,
        document_path: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        embedding_model: str,
        llm_model: str,
        openrouter_base_url: str,
    ):
        self.document_path = document_path
        self.top_k = top_k
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        self.loader_step = DocumentLoaderStep()
        self.chunker_step = DocumentChunkerStep(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever_builder_step = RetrieverBuilderStep(
            embedding_model=embedding_model,
            top_k=top_k,
        )
        self.context_step = ContextRetrieverStep()
        self.answer_step = AnswerGeneratorStep(
            llm_model=llm_model,
            base_url=openrouter_base_url,
        )
        self.retriever = self._prepare_retriever(chunk_size, chunk_overlap)

    def _prepare_retriever(self, chunk_size: int, chunk_overlap: int):
        print(f"\n[1/5] Loading document: {self.document_path}")
        documents = self.loader_step.load(self.document_path)

        print(f"[2/5] Chunking document (chunk_size={chunk_size}, overlap={chunk_overlap})")
        split_docs = self.chunker_step.chunk(documents)
        print(f"      → {len(split_docs)} chunks created")

        print(f"[3/5] Building retriever with embeddings '{self.embedding_model}'")
        return self.retriever_builder_step.build(split_docs)

    def answer(self, question: str) -> str:
        print(f"[4/5] Retrieving top {self.top_k} chunks")
        context = self.context_step.retrieve_context(self.retriever, question)

        print(f"[5/5] Generating answer with '{self.llm_model}'")
        return self.answer_step.generate(question, context)
