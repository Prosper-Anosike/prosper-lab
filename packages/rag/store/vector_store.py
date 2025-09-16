import os
from pathlib import Path
import faiss
import numpy as np
from typing import List
from openai import AzureOpenAI
from configs.settings import settings

class VectorStore:
    def __init__(self, index_dir: str, model: str = "text-embedding-ada-002"):
        print(f"[VectorStore] Initializing VectorStore with index_dir: {index_dir}, model: {model}")
        self.index_dir = Path(index_dir)
        self.model = model
        self.index_dir.mkdir(parents=True, exist_ok=True)
        print(f"[VectorStore] Created index directory: {self.index_dir}")
        self.index = faiss.IndexFlatL2(1536) 
        print(f"[VectorStore] Initialized FAISS index with dimension 1536")

    def load_chunks(self, input_dir: str) -> List[str]:
        """Load text chunks from the input directory."""
        print(f"[load_chunks] Loading chunks from directory: {input_dir}")
        input_path = Path(input_dir)
        print(f"[load_chunks] Input path exists: {input_path.exists()}")
        chunks = []
        
        chunk_files_found = 0
        for file in input_path.iterdir():
            print(f"[load_chunks] Checking file: {file.name}")
            if file.name.endswith('_chunks.txt'):  # Fixed: check filename ending, not suffix
                chunk_files_found += 1
                print(f"[load_chunks] Processing chunk file #{chunk_files_found}: {file.name}")
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    print(f"[load_chunks] File content length: {len(content)} characters")
                    if content:  # Only add non-empty chunks
                        # Split by double newlines (chunk separator)
                        file_chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                        print(f"[load_chunks] Found {len(file_chunks)} chunks in {file.name}")
                        chunks.extend(file_chunks)
                    else:
                        print(f"[load_chunks] File {file.name} is empty, skipping")
        
        print(f"[load_chunks] Total chunk files processed: {chunk_files_found}")
        print(f"[load_chunks] Total chunks loaded: {len(chunks)}")
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using Azure OpenAI API."""
        print(f"[generate_embeddings] Starting embedding generation for {len(chunks)} chunks")
        print(f"[generate_embeddings] Using model: {self.model}")
        
        if not chunks:
            print("[generate_embeddings] No chunks found for embedding generation.")
            return np.empty((0, 1536))
        
        try:
            azure_endpoint = "https://byupwai5996918872.openai.azure.com/"
            api_version = "2024-12-01-preview"
            api_key = settings.AZURE_OPENAI_API_KEY
            
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            )
            print(f"[generate_embeddings] Azure OpenAI client initialized successfully")
        except Exception as e:
            print(f"[generate_embeddings] Error initializing Azure OpenAI client: {e}")
            raise
        
        print(f"[generate_embeddings] Processing chunks...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            print(f"[generate_embeddings] Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
            try:
                embedding = client.embeddings.create(input=chunk, model=self.model).data[0].embedding
                embeddings.append(embedding)
                print(f"[generate_embeddings] Successfully generated embedding for chunk {i+1}")
            except Exception as e:
                print(f"[generate_embeddings] Error generating embedding for chunk {i+1}: {e}")
                raise
        
        result = np.vstack(embeddings)
        print(f"[generate_embeddings] Generated embeddings shape: {result.shape}")
        return result

    def save_index(self):
        """Save FAISS index to disk."""
        index_path = self.index_dir / "faiss_index"
        print(f"[save_index] Saving FAISS index to: {index_path}")
        print(f"[save_index] Index contains {self.index.ntotal} vectors")
        try:
            faiss.write_index(self.index, str(index_path))
            print(f"[save_index] Successfully saved index to disk")
        except Exception as e:
            print(f"[save_index] Error saving index: {e}")
            raise

    def load_index(self):
        """Load FAISS index from disk."""
        index_path = self.index_dir / "faiss_index"
        print(f"[load_index] Attempting to load index from: {index_path}")
        print(f"[load_index] Index file exists: {index_path.exists()}")
        
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                print(f"[load_index] Successfully loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"[load_index] Error loading index: {e}")
                raise
        else:
            print(f"[load_index] No existing index found, will create new one")

    def build_index(self, input_dir: str):
        """Build FAISS index from text chunks."""
        print(f"[build_index] Starting index building process")
        print(f"[build_index] Input directory: {input_dir}")
        
        chunks = self.load_chunks(input_dir)
        print(f"[build_index] Loaded {len(chunks)} chunks")
        
        if not chunks:
            print(f"[build_index] No chunks found, skipping embedding generation and index building")
            return
        
        embeddings = self.generate_embeddings(chunks)
        print(f"[build_index] Generated embeddings with shape: {embeddings.shape}")
        
        print(f"[build_index] Adding {len(embeddings)} embeddings to FAISS index")
        self.index.add(embeddings)
        print(f"[build_index] Index now contains {self.index.ntotal} vectors")
        
        self.save_index()
        print(f"[build_index] Index building process completed successfully")

if __name__ == "__main__":
    print(f"[main] Starting vector store script execution")
    input_dir = "data/index"
    index_dir = "data/index"
    print(f"[main] Input directory: {input_dir}")
    print(f"[main] Index directory: {index_dir}")
    
    store = VectorStore(index_dir)
    store.build_index(input_dir)
    print(f"[main] Script execution completed")
