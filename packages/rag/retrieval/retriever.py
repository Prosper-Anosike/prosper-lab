import faiss
import numpy as np
from pathlib import Path
from typing import List
from openai import AzureOpenAI
from configs.settings import settings
# from llama_index import SimpleKeywordTableIndex  # TODO: Fix llama_index import

class Retriever:
    def __init__(self, index_dir: str, top_k: int = 5):
        print(f"[Retriever] Initializing Retriever with index_dir: {index_dir}, top_k: {top_k}")
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_index"
        self.top_k = top_k
        self.model = "text-embedding-ada-002"  # Use same model as VectorStore
        print(f"[Retriever] Using embedding model: {self.model}")
        self.index = self.load_index()

    def load_index(self):
        """Load FAISS index from disk."""
        print(f"[load_index] Attempting to load index from: {self.index_path}")
        print(f"[load_index] Index file exists: {self.index_path.exists()}")
        
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                print(f"[load_index] Successfully loaded index with {index.ntotal} vectors")
                return index
            except Exception as e:
                print(f"[load_index] Error loading index: {e}")
                raise
        else:
            print(f"[load_index] FAISS index not found at {self.index_path}")
            raise FileNotFoundError("FAISS index not found.")

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query using Azure OpenAI API."""
        print(f"[generate_query_embedding] Generating embedding for query: '{query[:50]}...'")
        print(f"[generate_query_embedding] Using model: {self.model}")
        
        print(f"[generate_query_embedding] Azure OpenAI API key set: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
        
        try:
            # Azure OpenAI configuration
            azure_endpoint = "https://byupwai5996918872.openai.azure.com/"
            api_version = "2024-12-01-preview"
            api_key = settings.OPENAI_API_KEY
            
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            )
            print(f"[generate_query_embedding] Azure OpenAI client initialized successfully")
            
            print(f"[generate_query_embedding] Making embedding API call...")
            embedding = client.embeddings.create(input=query, model=self.model).data[0].embedding
            result = np.array([embedding])  # Shape: (1, 1536)
            print(f"[generate_query_embedding] Generated embedding shape: {result.shape}")
            return result
        except Exception as e:
            print(f"[generate_query_embedding] Error generating embedding: {e}")
            print(f"[generate_query_embedding] Error type: {type(e).__name__}")
            raise

    def retrieve(self, query_embedding: np.ndarray) -> List[int]:
        """Retrieve top-k relevant chunks based on query embedding."""
        print(f"[retrieve] Searching for top-{self.top_k} similar chunks")
        print(f"[retrieve] Query embedding shape: {query_embedding.shape}")
        print(f"[retrieve] Index contains {self.index.ntotal} vectors")
        
        try:
            distances, indices = self.index.search(query_embedding, self.top_k)
            print(f"[retrieve] Search completed. Distances: {distances[0]}")
            print(f"[retrieve] Found indices: {indices[0]}")
            return indices.flatten().tolist()
        except Exception as e:
            print(f"[retrieve] Error during search: {e}")
            raise

    def load_all_chunks(self, chunk_dir: str) -> List[str]:
        """Load all chunks from _chunks.txt files in the directory."""
        print(f"[load_all_chunks] Loading all chunks from directory: {chunk_dir}")
        chunk_path = Path(chunk_dir)
        all_chunks = []
        
        for file in chunk_path.iterdir():
            if file.name.endswith('_chunks.txt'):
                print(f"[load_all_chunks] Processing chunk file: {file.name}")
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # Split by double newlines (chunk separator)
                        file_chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                        print(f"[load_all_chunks] Found {len(file_chunks)} chunks in {file.name}")
                        all_chunks.extend(file_chunks)
        
        print(f"[load_all_chunks] Total chunks loaded: {len(all_chunks)}")
        return all_chunks

    def fetch_chunks(self, indices: List[int], chunk_dir: str) -> List[str]:
        """Fetch chunks corresponding to indices."""
        print(f"[fetch_chunks] Fetching chunks for indices: {indices}")
        print(f"[fetch_chunks] Chunk directory: {chunk_dir}")
        
        # Load all chunks first
        all_chunks = self.load_all_chunks(chunk_dir)
        
        chunks = []
        chunks_found = 0
        
        for idx in indices:
            print(f"[fetch_chunks] Looking for chunk at index: {idx}")
            
            if 0 <= idx < len(all_chunks):
                chunk_content = all_chunks[idx]
                chunks.append(chunk_content)
                chunks_found += 1
                print(f"[fetch_chunks] Successfully loaded chunk {idx} (length: {len(chunk_content)} chars)")
            else:
                print(f"[fetch_chunks] Index {idx} is out of range (total chunks: {len(all_chunks)})")
        
        print(f"[fetch_chunks] Successfully fetched {chunks_found}/{len(indices)} chunks")
        return chunks

    def orchestrate_retrieval(self, query: str, chunk_dir: str) -> List[str]:
        """Orchestrate retrieval process."""
        print(f"[orchestrate_retrieval] Starting retrieval for query: '{query[:50]}...'")
        print(f"[orchestrate_retrieval] Chunk directory: {chunk_dir}")
        
        # Generate proper embedding for the query
        query_embedding = self.generate_query_embedding(query)
        
        # Retrieve similar chunk indices
        indices = self.retrieve(query_embedding)
        
        # Fetch the actual chunk content
        chunks = self.fetch_chunks(indices, chunk_dir)
        
        print(f"[orchestrate_retrieval] Retrieval completed. Found {len(chunks)} chunks")
        return chunks

if __name__ == "__main__":
    index_dir = "data/index"
    chunk_dir = "data/index"
    retriever = Retriever(index_dir)
    query = "What is the process for document ingestion?"
    results = retriever.orchestrate_retrieval(query, chunk_dir)
    print(results)
