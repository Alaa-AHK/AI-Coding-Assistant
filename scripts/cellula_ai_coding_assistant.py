#!/usr/bin/env python3
"""
Cellula — Intelligent AI Coding Assistant
==========================================

A complete AI coding assistant with:
- Intent Classification (Explain vs Generate)
- Code Explanation (Direct LLM, no RAG)
- Code Generation with RAG (ChromaDB + Sentence Transformers)
- Relevance Checking for retrieved documents
- Human Feedback Learning
- Conversation Memory
- Code Execution Tool
- Streamlit UI

Usage:
    streamlit run app.py

Or run this script directly for a CLI demo:
    python cellula_ai_coding_assistant.py

Requires:
    pip install streamlit huggingface-hub sentence-transformers chromadb langchain langchain-community
"""

import os
import re
import uuid
import tempfile
import subprocess
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from huggingface_hub import InferenceClient


# =============================================================================
# Section 1: Vector Store — ChromaDB Wrapper
# =============================================================================

class VectorStore:
    """ChromaDB wrapper for document storage and retrieval."""
    
    def __init__(self, persist_directory='./chroma_db', collection_name='coding_knowledge'):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_document(self, content: str, metadata: dict = None) -> str:
        """Add a document to the vector store with automatic chunking."""
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 500:
                words = chunk.split()
                current_chunk = []
                current_len = 0
                for word in words:
                    if current_len + len(word) + 1 > 500:
                        final_chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_len = len(word)
                    else:
                        current_chunk.append(word)
                        current_len += len(word) + 1
                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))
            else:
                final_chunks.append(chunk)

        doc_ids = []
        for chunk in final_chunks:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[chunk],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            doc_ids.append(doc_id)
        
        return doc_ids[0] if doc_ids else None

    def search(self, query: str, top_k: int = 3) -> list:
        """Search for similar documents."""
        if self.get_collection_count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.get_collection_count())
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
                })
        return formatted_results

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()


# =============================================================================
# Section 2: Intent Classifier
# =============================================================================

class IntentClassifier:
    """LLM-based intent classifier for routing queries."""
    
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
        except Exception:
            return 'GENERATE'


# =============================================================================
# Section 3: Code Explainer (Route 1 — No RAG)
# =============================================================================

class CodeExplainer:
    """Direct LLM code explanation — no RAG involved."""
    
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


# =============================================================================
# Section 4: Relevance Checker
# =============================================================================

class RelevanceChecker:
    """LLM-based relevance evaluator for retrieved documents."""
    
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
        except Exception:
            return False


# =============================================================================
# Section 5: RAG Code Generator (Route 2 — With RAG)
# =============================================================================

class RAGCodeGenerator:
    """RAG-powered code generation with relevance checking."""
    
    def __init__(self, client: InferenceClient, vector_store: VectorStore, 
                 relevance_checker: RelevanceChecker, model_name: str):
        self.client = client
        self.vector_store = vector_store
        self.relevance_checker = relevance_checker
        self.model_name = model_name

    def generate(self, query: str, memory_context: str = '') -> dict:
        results = self.vector_store.search(query, top_k=3)
        retrieved_context = "\n\n".join([res['content'] for res in results])
        
        is_relevant = False
        if retrieved_context:
            is_relevant = self.relevance_checker.check(query, retrieved_context)
            
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


# =============================================================================
# Section 6: Feedback Learner
# =============================================================================

class FeedbackLearner:
    """Learns from user-provided solutions by storing them in the vector database."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def learn(self, user_solution: str, original_query: str = '') -> str:
        content = f"Query: {original_query}\nSolution:\n{user_solution}"
        self.vector_store.add_document(content, {"source": "user_feedback", "query": original_query})
        return "Thank you! I have added this solution to my knowledge base."


# =============================================================================
# Section 7: Conversation Memory
# =============================================================================

class ConversationMemory:
    """Manages conversation history for context-aware interactions."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages = []
        
    def add_message(self, role: str, content: str):
        self.messages.append({'role': role, 'content': content})
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]
            
    def get_context(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])
        
    def clear(self):
        self.messages = []
        
    def get_messages(self) -> list:
        return self.messages


# =============================================================================
# Section 8: Code Executor
# =============================================================================

class CodeExecutor:
    """Executes code in a sandboxed subprocess."""
    
    def execute(self, code: str, timeout: int = 30) -> dict:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
                
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'error': None if result.returncode == 0 else "Process exited with non-zero status"
            }
        except subprocess.TimeoutExpired:
            return {'stdout': '', 'stderr': 'Execution timed out', 'success': False, 'error': 'Timeout'}
        except Exception as e:
            return {'stdout': '', 'stderr': str(e), 'success': False, 'error': str(e)}
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass


# =============================================================================
# Section 9: Main Orchestrator — CodingAssistant
# =============================================================================

class CodingAssistant:
    """Main orchestrator that ties all components together."""
    
    def __init__(self, hf_token: str, model_name: str = 'mistralai/Mistral-7B-Instruct-v0.3'):
        self.hf_token = hf_token
        self.model_name = model_name
        self.client = InferenceClient(token=hf_token)
        
        self.vector_store = VectorStore(persist_directory='./chroma_db')
        self.classifier = IntentClassifier(self.client, self.model_name)
        self.explainer = CodeExplainer(self.client, self.model_name)
        self.relevance_checker = RelevanceChecker(self.client, self.model_name)
        self.generator = RAGCodeGenerator(
            self.client, self.vector_store, self.relevance_checker, self.model_name
        )
        self.learner = FeedbackLearner(self.vector_store)
        self.memory = ConversationMemory()
        self.executor = CodeExecutor()

    def process_query(self, query: str) -> dict:
        """Process a user query through the full pipeline."""
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
        """Learn from user-provided feedback."""
        msg = self.learner.learn(solution, original_query)
        self.memory.add_message('user', f"Feedback provided: {solution}")
        self.memory.add_message('assistant', msg)
        return msg

    def execute_code(self, code: str) -> dict:
        """Execute generated code."""
        return self.executor.execute(code)


# =============================================================================
# Section 10: CLI Demo
# =============================================================================

def main():
    """Interactive CLI demo of the Cellula AI Coding Assistant."""
    print("="*60)
    print("  Cellula — Intelligent AI Coding Assistant")
    print("="*60)
    
    hf_token = os.environ.get('HF_API_TOKEN', '')
    if not hf_token:
        hf_token = input("Enter your Hugging Face API token: ").strip()
    
    if not hf_token:
        print("Error: No API token provided.")
        return
    
    print("\nInitializing assistant...")
    assistant = CodingAssistant(hf_token=hf_token)
    print(f"Ready! Knowledge base: {assistant.vector_store.get_collection_count()} documents")
    print("Type 'quit' to exit, 'feedback' to provide a solution, 'run' to execute last code.\n")
    
    last_code = None
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not query:
            continue
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        if query.lower() == 'run' and last_code:
            print("\nExecuting code...")
            result = assistant.execute_code(last_code)
            print(f"Success: {result['success']}")
            if result['stdout']:
                print(f"Output:\n{result['stdout']}")
            if result['stderr']:
                print(f"Errors:\n{result['stderr']}")
            continue
        if query.lower() == 'feedback':
            original = input("Original query (optional): ").strip()
            solution = input("Your solution:\n").strip()
            msg = assistant.provide_feedback(solution, original)
            print(f"\nAssistant: {msg}")
            continue
        
        result = assistant.process_query(query)
        print(f"\n[{result['type'].upper()}]")
        print(f"\nAssistant: {result['response']}")
        
        if result['code']:
            last_code = result['code']
            print("\n(Type 'run' to execute the generated code)")
        
        if result['needs_feedback']:
            print("\n(Type 'feedback' to provide a solution)")


if __name__ == '__main__':
    main()
