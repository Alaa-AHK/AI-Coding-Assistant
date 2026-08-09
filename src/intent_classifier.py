"""
IntentClassifier — Uses a LangChain LCEL chain to classify user intent.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class IntentClassifier:
    """Classifies user intent as EXPLAIN or GENERATE using a LangChain chain."""

    def __init__(self, chat_model):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system",
                 'You are an intent classifier. Respond with ONLY the word '
                 '"EXPLAIN" if the user wants code explained, or "GENERATE" '
                 'if the user wants code generated/written. Do not add any other text.'),
                ("human", "{query}")
            ])
            | chat_model.bind(max_tokens=10)
            | StrOutputParser()
        )

    def classify(self, query: str) -> str:
        try:
            result = self.chain.invoke({"query": query}).strip().upper()
            if 'EXPLAIN' in result:
                return 'EXPLAIN'
            return 'GENERATE'
        except Exception as e:
            print(f"API Error in IntentClassifier: {e}")
            return 'GENERATE'
