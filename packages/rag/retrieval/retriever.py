import faiss
import numpy as np
from pathlib import Path
from typing import List
from llama_index import SimpleKeywordTableIndex

class Retriever:
    def __init__(self, index_dir: str, top_k: int = 5):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_index"
        self.top_k = top_k
        self.index = self.load_index()

    def load_index(self):
        """Load FAISS index from disk."""
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        else:
            raise FileNotFoundError("FAISS index not found.")

    def retrieve(self, query_embedding: np.ndarray) -> List[int]:
        """Retrieve top-k relevant chunks based on query embedding."""
        distances, indices = self.index.search(query_embedding, self.top_k)
        return indices.flatten().tolist()

    def fetch_chunks(self, indices: List[int], chunk_dir: str) -> List[str]:
        """Fetch chunks corresponding to indices."""
        chunk_path = Path(chunk_dir)
        chunks = []
        for idx in indices:
            chunk_file = chunk_path / f"chunk_{idx}.txt"
            if chunk_file.exists():
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunks.append(f.read())
        return chunks

    def orchestrate_retrieval(self, query: str, chunk_dir: str) -> List[str]:
        """Orchestrate retrieval using LlamaIndex."""
        query_embedding = np.array([query])  # Replace with actual embedding generation
        indices = self.retrieve(query_embedding)
        chunks = self.fetch_chunks(indices, chunk_dir)
        return chunks

if __name__ == "__main__":
    index_dir = "data/index"
    chunk_dir = "data/index"
    retriever = Retriever(index_dir)
    query = "What is the process for document ingestion?"
    results = retriever.orchestrate_retrieval(query, chunk_dir)
    print(results)
