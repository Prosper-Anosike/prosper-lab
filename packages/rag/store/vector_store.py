import os
from pathlib import Path
import faiss
import numpy as np
from typing import List
import openai
from configs.settings import settings

class VectorStore:
    def __init__(self, index_dir: str, model: str = "text-embedding-3-large"):
        self.index_dir = Path(index_dir)
        self.model = model
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(1536)  # Assuming embedding size is 1536

    def load_chunks(self, input_dir: str) -> List[str]:
        """Load text chunks from the input directory."""
        input_path = Path(input_dir)
        chunks = []
        for file in input_path.iterdir():
            if file.suffix == '_chunks.txt':
                with open(file, 'r', encoding='utf-8') as f:
                    chunks.extend(f.readlines())
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using OpenAI API."""
        openai.api_key = settings.OPENAI_API_KEY
        if not chunks:
            print("No chunks found for embedding generation.")
            return np.empty((0, 1536))
        embeddings = [
            openai.embeddings.create(input=chunk, model=self.model).data[0].embedding
            for chunk in chunks
        ]
        return np.vstack(embeddings)

    def save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(self.index_dir / "faiss_index"))

    def load_index(self):
        """Load FAISS index from disk."""
        index_path = self.index_dir / "faiss_index"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

    def build_index(self, input_dir: str):
        """Build FAISS index from text chunks."""
        chunks = self.load_chunks(input_dir)
        embeddings = self.generate_embeddings(chunks)
        self.index.add(embeddings)
        self.save_index()

if __name__ == "__main__":
    input_dir = "data/index"
    index_dir = "data/index"
    store = VectorStore(index_dir)
    store.build_index(input_dir)
