"""Step 5: generate final answer from context and question."""

import os


class AnswerGeneratorStep:
    """Build and run the LangChain answer generation chain."""

    def __init__(self, llm_model: str, base_url: str):
        self.llm_model = llm_model
        self.base_url = base_url

    def _build_llm(self):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required. Install with: pip install langchain-openai"
            ) from exc

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Get a free key at https://openrouter.ai/"
            )

        return ChatOpenAI(
            model=self.llm_model,
            api_key=api_key,
            base_url=self.base_url,
            temperature=0.2,
        )

    def generate(self, question: str, context: str) -> str:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
        except ImportError as exc:
            raise ImportError(
                "LangChain core is missing. Install with: pip install langchain"
            ) from exc

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = prompt | self._build_llm() | StrOutputParser()
        return chain.invoke({"context": context, "question": question}).strip()
