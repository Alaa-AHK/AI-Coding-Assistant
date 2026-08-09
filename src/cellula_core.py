"""
Cellula Core — Main orchestrator that ties all LangChain-powered components together.
"""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from src.vector_store import VectorStore
from src.intent_classifier import IntentClassifier
from src.code_explainer import CodeExplainer
from src.relevance_checker import RelevanceChecker
from src.rag_generator import RAGCodeGenerator
from src.feedback_learner import FeedbackLearner
from src.conversation_memory import ConversationMemory
from src.code_executor import CodeExecutor


class CodingAssistant:
    """Main orchestrator that ties all LangChain-powered components together."""

    def __init__(self, hf_token: str, model_name: str = 'Qwen/Qwen2.5-7B-Instruct'):
        self.hf_token = hf_token
        self.model_name = model_name

        # LangChain LLM: HuggingFace Inference API via LangChain wrapper
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=1500,
        )
        self.chat_model = ChatHuggingFace(llm=llm)

        # LangChain-powered vector store
        self.vector_store = VectorStore(
            persist_directory='./chroma_db',
            collection_name='coding_knowledge'
        )

        # All components now receive the LangChain chat model
        self.classifier = IntentClassifier(self.chat_model)
        self.explainer = CodeExplainer(self.chat_model)
        self.relevance_checker = RelevanceChecker(self.chat_model)
        self.generator = RAGCodeGenerator(self.chat_model, self.vector_store, self.relevance_checker)
        self.learner = FeedbackLearner(self.vector_store)
        self.memory = ConversationMemory()
        self.executor = CodeExecutor()

    def process_query(self, query: str) -> dict:
        self.memory.add_message('user', query)
        memory_context = self.memory.get_context()

        intent = self.classifier.classify(query)

        if intent == 'EXPLAIN':
            explanation = self.explainer.explain(query, memory_context)
            self.memory.add_message('assistant', explanation)
            return {
                'type': 'explain',
                'response': explanation,
                'code': None,
                'needs_feedback': False
            }
        else:
            result = self.generator.generate(query, memory_context)
            self.memory.add_message('assistant', result['message'])
            return {
                'type': 'generate',
                'response': result['message'],
                'code': result['code'],
                'needs_feedback': result['needs_feedback']
            }

    def provide_feedback(self, solution: str, original_query: str = '') -> str:
        msg = self.learner.learn(solution, original_query)
        self.memory.add_message('user', f"Feedback provided: {solution}")
        self.memory.add_message('assistant', msg)
        return msg

    def execute_code(self, code: str) -> dict:
        return self.executor.execute(code)
