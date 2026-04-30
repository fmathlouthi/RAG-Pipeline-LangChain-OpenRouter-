"""Step 4: retrieve relevant chunks and build context text."""


class ContextRetrieverStep:
    """Convert retrieved docs into a single context string."""

    @staticmethod
    def retrieve_context(retriever, question: str) -> str:
        docs = retriever.invoke(question)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
