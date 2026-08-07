import uuid
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class VectorStore:
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
        # Simple paragraph splitting for chunking
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        # Further split chunks if they are > 500 chars (approximate)
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

    def search(self, query: str, top_k: int = 3) -> list[dict]:
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
        return self.collection.count()
