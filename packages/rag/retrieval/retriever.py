import json
import faiss
import numpy as np
from pathlib import Path
from typing import List
from openai import AzureOpenAI
from configs.settings import settings

class Retriever:
    def __init__(self, index_dir: str, top_k: int = 5):
        print(f"[Retriever] Initializing Retriever with index_dir: {index_dir}, top_k: {top_k}")
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_index"
        self.mapping_path = self.index_dir / "chunks.json"
        self.top_k = top_k
        self.model = settings.AZURE_EMBED_DEPLOYMENT
        print(f"[Retriever] Using embedding deployment: {self.model}")
        self.index = self.load_index()

    def load_index(self):
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
        print(f"[generate_query_embedding] Generating embedding for query: '{query[:50]}...'")
        print(f"[generate_query_embedding] Using deployment: {self.model}")
        print(f"[generate_query_embedding] Azure OpenAI API key set: {'Yes' if settings.AZURE_OPENAI_API_KEY else 'No'}")
        try:
            client = AzureOpenAI(
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
            )
            print(f"[generate_query_embedding] Azure OpenAI client initialized successfully")
            embedding = client.embeddings.create(input=query, model=self.model).data[0].embedding
            result = np.array([embedding], dtype="float32")
            print(f"[generate_query_embedding] Generated embedding shape: {result.shape}")
            return result
        except Exception as e:
            print(f"[generate_query_embedding] Error generating embedding: {e}")
            print(f"[generate_query_embedding] Error type: {type(e).__name__}")
            raise

    def retrieve(self, query_embedding: np.ndarray) -> List[int]:
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
        print(f"[load_all_chunks] Loading all chunks from directory: {chunk_dir}")
        chunk_path = Path(chunk_dir)
        all_chunks: List[str] = []
        for file in sorted(chunk_path.iterdir(), key=lambda p: p.name):
            if file.name.endswith("_chunks.txt"):
                print(f"[load_all_chunks] Processing chunk file: {file.name}")
                content = file.read_text(encoding="utf-8").strip()
                if content:
                    file_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                    print(f"[load_all_chunks] Found {len(file_chunks)} chunks in {file.name}")
                    all_chunks.extend(file_chunks)
        print(f"[load_all_chunks] Total chunks loaded: {len(all_chunks)}")
        return all_chunks

    def fetch_chunks(self, indices: List[int], chunk_dir: str) -> List[str]:
        print(f"[fetch_chunks] Fetching chunks for indices: {indices}")
        print(f"[fetch_chunks] Chunk directory: {chunk_dir}")
        if self.mapping_path.exists():
            print(f"[fetch_chunks] Using mapping file: {self.mapping_path}")
            all_chunks = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        else:
            print("[fetch_chunks] Mapping file not found, falling back to scanning chunk_dir")
            all_chunks = self.load_all_chunks(chunk_dir)
        chunks = []
        for idx in indices:
            if 0 <= idx < len(all_chunks):
                chunk_content = all_chunks[idx]
                chunks.append(chunk_content)
                print(f"[fetch_chunks] Loaded chunk {idx} (len: {len(chunk_content)})")
            else:
                print(f"[fetch_chunks] Index {idx} out of range (total chunks: {len(all_chunks)})")
        print(f"[fetch_chunks] Successfully fetched {len(chunks)}/{len(indices)} chunks")
        return chunks

    def orchestrate_retrieval(self, query: str, chunk_dir: str) -> List[str]:
        print(f"[orchestrate_retrieval] Starting retrieval for query: '{query[:50]}...'")
        query_embedding = self.generate_query_embedding(query)
        indices = self.retrieve(query_embedding)
        chunks = self.fetch_chunks(indices, chunk_dir)
        print(f"[orchestrate_retrieval] Retrieval completed. Found {len(chunks)} chunks")
        return chunks

if __name__ == "__main__":
    index_dir = "data/index"
    chunk_dir = "data/chunks"
    retriever = Retriever(index_dir)
    query = "What is the process for document ingestion?"
    results = retriever.orchestrate_retrieval(query, chunk_dir)
    print(results)
