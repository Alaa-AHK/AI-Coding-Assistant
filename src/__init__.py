"""
Cellula — Intelligent AI Coding Assistant
Source package containing all core modules.
"""

from src.cellula_core import (
    CodingAssistant,
    IntentClassifier,
    CodeExplainer,
    RelevanceChecker,
    RAGCodeGenerator,
    FeedbackLearner,
    ConversationMemory,
    CodeExecutor,
)
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
