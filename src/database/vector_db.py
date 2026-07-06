import os
import numpy as np
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from google import genai
from src.config import Config

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB utilizing the new google-genai SDK.
    Falls back to mock embeddings if API key is not present.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        if api_key:
            # Initialize using the new Google GenAI SDK
            self.client = genai.Client(api_key=api_key)
            print("Gemini GenAI Embedding Function initialized.")
        else:
            self.client = None
            print("WARNING: Gemini API Key not found. Using deterministic mock embedding function.")

    def __call__(self, input: Documents) -> Embeddings:
        if self.client:
            try:
                # Use new SDK embedding method
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=list(input)
                )
                return [emb.values for emb in response.embeddings]
            except Exception as e:
                print(f"Failed to generate embeddings from Gemini API: {e}. Falling back to mock vectors.")
                
        # Mock Embedding: Return a normalized deterministic random vector based on document content
        embeddings = []
        for doc in input:
            # Deterministic seed using characters in the text
            seed = sum(ord(c) for c in doc[:20]) + len(doc)
            rng = np.random.default_rng(seed)
            vec = rng.normal(0, 1, 768)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec.tolist())
        return embeddings

class VectorDB:
    def __init__(self):
        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_function = GeminiEmbeddingFunction(Config.GEMINI_API_KEY)
        
    def get_or_create_collection(self, name: str):
        """Gets or creates a ChromaDB collection with the Gemini embedding function."""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function
        )
        
    def add_documents(self, collection_name: str, documents: list, metadatas: list, ids: list):
        """Adds text documents, metadata, and unique IDs to a collection."""
        collection = self.get_or_create_collection(collection_name)
        # ChromaDB handles generating embeddings via embedding_function
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    def query(self, collection_name: str, query_text: str, n_results: int = 3):
        """Queries the vector database for nearest neighbors."""
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
