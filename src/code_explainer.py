from huggingface_hub import InferenceClient


class CodeExplainer:
    """Explains code using the LLM without RAG retrieval."""

    def __init__(self, client: InferenceClient, model_name: str):
        self.client = client
        self.model_name = model_name

    def explain(self, code: str, memory_context: str = '') -> str:
        messages = [
            {'role': 'system', 'content': 'You are an expert Python developer. Explain the provided code clearly with line-by-line analysis where appropriate.'},
            {'role': 'user', 'content': f'Context:\n{memory_context}\n\nCode to explain:\n{code}'}
        ]
        response = self.client.chat_completion(
            model=self.model_name,
            messages=messages,
            max_tokens=1024
        )
        return response.choices[0].message.content
