from huggingface_hub import InferenceClient


class IntentClassifier:
    """Classifies user intent as EXPLAIN or GENERATE."""

    def __init__(self, client: InferenceClient, model_name: str):
        self.client = client
        self.model_name = model_name

    def classify(self, query: str) -> str:
        messages = [
            {'role': 'system', 'content': 'You are an intent classifier. Respond with ONLY the word "EXPLAIN" if the user wants code explained, or "GENERATE" if the user wants code generated/written. Do not add any other text.'},
            {'role': 'user', 'content': query}
        ]
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            if 'EXPLAIN' in result:
                return 'EXPLAIN'
            return 'GENERATE'
        except Exception as e:
            print(f"API Error in IntentClassifier: {e}")
            return 'GENERATE'  # Default to generate
