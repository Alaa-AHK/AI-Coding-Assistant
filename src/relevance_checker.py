"""
RelevanceChecker — Uses a LangChain LCEL chain to verify context relevance.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RelevanceChecker:
    """Checks whether retrieved context is relevant to the user query using a LangChain chain."""

    def __init__(self, chat_model):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system",
                 'You are a relevance checker. Given a query and some context, '
                 'respond with ONLY "TRUE" if the context contains information '
                 'relevant to answering the query, or "FALSE" if it does not.'),
                ("human",
                 "Query: {query}\n\nContext:\n{context}")
            ])
            | chat_model.bind(max_tokens=10)
            | StrOutputParser()
        )

    def check(self, query: str, retrieved_context: str) -> bool:
        try:
            result = self.chain.invoke({
                "query": query,
                "context": retrieved_context
            }).strip().upper()
            return 'TRUE' in result
        except Exception as e:
            print(f"API Error in RelevanceChecker: {e}")
            return False
