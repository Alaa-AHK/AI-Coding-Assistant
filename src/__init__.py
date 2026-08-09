"""
Cellula — Intelligent AI Coding Assistant
Source package containing all core modules.
"""

from src.cellula_core import CodingAssistant
from src.intent_classifier import IntentClassifier
from src.code_explainer import CodeExplainer
from src.relevance_checker import RelevanceChecker
from src.rag_generator import RAGCodeGenerator
from src.feedback_learner import FeedbackLearner
from src.conversation_memory import ConversationMemory
from src.code_executor import CodeExecutor
from src.vector_store import VectorStore

__all__ = [
    "CodingAssistant",
    "IntentClassifier",
    "CodeExplainer",
    "RelevanceChecker",
    "RAGCodeGenerator",
    "FeedbackLearner",
    "ConversationMemory",
    "CodeExecutor",
    "VectorStore",
]
