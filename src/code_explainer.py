"""
CodeExplainer — Uses a LangChain LCEL chain to explain code.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class CodeExplainer:
    """Explains code using the LLM through a LangChain chain (no RAG retrieval)."""

    def __init__(self, chat_model):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system",
                 'You are an expert Python developer. Explain the provided code '
                 'clearly with line-by-line analysis where appropriate.'),
                ("human",
                 "Context:\n{memory_context}\n\nCode to explain:\n{code}")
            ])
            | chat_model.bind(max_tokens=1024)
            | StrOutputParser()
        )

    def explain(self, code: str, memory_context: str = '') -> str:
        return self.chain.invoke({
            "code": code,
            "memory_context": memory_context
        })
