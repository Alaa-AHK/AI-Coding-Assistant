import re
from huggingface_hub import InferenceClient
from src.vector_store import VectorStore
from src.relevance_checker import RelevanceChecker


class RAGCodeGenerator:
    """Generates code using RAG pipeline with vector store retrieval."""

    def __init__(self, client: InferenceClient, vector_store: VectorStore, relevance_checker: RelevanceChecker, model_name: str):
        self.client = client
        self.vector_store = vector_store
        self.relevance_checker = relevance_checker
        self.model_name = model_name

    def generate(self, query: str, memory_context: str = '') -> dict:
        results = self.vector_store.search(query, top_k=3)
        print(f"[RAG] Searched database. Found {len(results)} documents.")

        retrieved_context = "\n\n".join([res['content'] for res in results])

        is_relevant = False
        if retrieved_context:
            is_relevant = self.relevance_checker.check(query, retrieved_context)
            print(f"[Relevance] AI checked the documents and said: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")

        if not is_relevant:
            return {
                'code': None,
                'explanation': None,
                'needs_feedback': True,
                'message': "I don't have enough information in my knowledge base to answer this query. Could you please provide a solution or more context so I can learn from it?"
            }

        messages = [
            {'role': 'system', 'content': 'You are an AI coding assistant. Generate Python code to answer the user query. Use the provided context if helpful. Provide the code in a markdown block, and optionally a brief explanation.'},
            {'role': 'user', 'content': f'Conversation Context:\n{memory_context}\n\nKnowledge Base Context:\n{retrieved_context}\n\nQuery: {query}'}
        ]

        response = self.client.chat_completion(
            model=self.model_name,
            messages=messages,
            max_tokens=1500
        )
        content = response.choices[0].message.content

        # Extract code if present
        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)

        code_block = code_match.group(1) if code_match else None

        return {
            'code': code_block,
            'explanation': content,
            'needs_feedback': False,
            'message': content
        }
