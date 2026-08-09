import re
import tempfile
import subprocess
from huggingface_hub import InferenceClient
from src.vector_store import VectorStore

class IntentClassifier:
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
            print(f"⚠️ API Error in IntentClassifier: {e}")
            return 'GENERATE'  # Default to generate

class CodeExplainer:
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

class RelevanceChecker:
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
            print(f"⚠️ API Error in RelevanceChecker: {e}")
            return False

class RAGCodeGenerator:
    def __init__(self, client: InferenceClient, vector_store: VectorStore, relevance_checker: RelevanceChecker, model_name: str):
        self.client = client
        self.vector_store = vector_store
        self.relevance_checker = relevance_checker
        self.model_name = model_name

    def generate(self, query: str, memory_context: str = '') -> dict:
        results = self.vector_store.search(query, top_k=3)
        print(f"🔍 [RAG] Searched database. Found {len(results)} documents.")
        
        retrieved_context = "\n\n".join([res['content'] for res in results])
        
        is_relevant = False
        if retrieved_context:
            is_relevant = self.relevance_checker.check(query, retrieved_context)
            print(f"🧠 [Relevance] AI checked the documents and said: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
            
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

class FeedbackLearner:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def learn(self, user_solution: str, original_query: str = '') -> str:
        content = f"Query: {original_query}\nSolution:\n{user_solution}"
        self.vector_store.add_document(content, {"source": "user_feedback", "query": original_query})
        return "Thank you! I have added this solution to my knowledge base."

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
            return {
                'stdout': '',
                'stderr': 'Execution timed out',
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'error': str(e)
            }
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

class CodingAssistant:
    """Main orchestrator that ties all components together."""
    
    def __init__(self, hf_token: str, model_name: str = 'Qwen/Qwen2.5-7B-Instruct'):
        self.hf_token = hf_token
        self.model_name = model_name
        self.client = InferenceClient(token=hf_token)
        
        self.vector_store = VectorStore(
            persist_directory='./chroma_db',
            collection_name='coding_knowledge'
        )
        
        self.classifier = IntentClassifier(self.client, self.model_name)
        self.explainer = CodeExplainer(self.client, self.model_name)
        self.relevance_checker = RelevanceChecker(self.client, self.model_name)
        self.generator = RAGCodeGenerator(self.client, self.vector_store, self.relevance_checker, self.model_name)
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
