from huggingface_hub import InferenceClient


class RelevanceChecker:
    """Checks whether retrieved context is relevant to the user query."""

    def __init__(self, client: InferenceClient, model_name: str):
        self.client = client
        self.model_name = model_name

    def check(self, query: str, retrieved_context: str) -> bool:
        messages = [
            {'role': 'system', 'content': 'You are a relevance checker. Given a query and some context, respond with ONLY "TRUE" if the context contains information relevant to answering the query, or "FALSE" if it does not.'},
            {'role': 'user', 'content': f'Query: {query}\n\nContext:\n{retrieved_context}'}
        ]
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            return 'TRUE' in result
        except Exception as e:
            print(f"API Error in RelevanceChecker: {e}")
            return False
