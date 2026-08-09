"""
VectorStore — LangChain-powered ChromaDB wrapper for document storage and retrieval.
Uses HuggingFaceEmbeddings and RecursiveCharacterTextSplitter from LangChain.
"""
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:
    """ChromaDB vector store wrapped with LangChain for embedding, chunking, and retrieval."""

    def __init__(self, persist_directory='./chroma_db', collection_name='coding_knowledge'):
        # LangChain embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # LangChain Chroma vector store
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        # LangChain text splitter for intelligent chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

    def add_document(self, content: str, metadata: dict = None) -> str:
        """Split content into chunks and add to the vector store."""
        chunks = self.text_splitter.split_text(content)
        ids = self.db.add_texts(
            texts=chunks,
            metadatas=[metadata or {} for _ in chunks]
        )
        return ids[0] if ids else None

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search the vector store for relevant documents."""
        if self.get_collection_count() == 0:
            return []

        results = self.db.similarity_search_with_score(
            query, k=min(top_k, self.get_collection_count())
        )

        formatted = []
        for doc, score in results:
            formatted.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'distance': score
            })
        return formatted

    def as_retriever(self, top_k: int = 3):
        """Return a LangChain Retriever object for use in chains."""
        return self.db.as_retriever(search_kwargs={"k": top_k})

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.db._collection.count()
