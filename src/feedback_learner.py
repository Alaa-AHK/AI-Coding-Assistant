from src.vector_store import VectorStore


class FeedbackLearner:
    """Learns from user-provided solutions by storing them in the vector database."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def learn(self, user_solution: str, original_query: str = '') -> str:
        content = f"Query: {original_query}\nSolution:\n{user_solution}"
        self.vector_store.add_document(content, {"source": "user_feedback", "query": original_query})
        return "Thank you! I have added this solution to my knowledge base."
