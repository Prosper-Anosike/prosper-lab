import os
from pathlib import Path
import faiss
import numpy as np
from typing import List
from openai import AzureOpenAI
from configs.settings import settings
from utils.RAGLogger import RAGLogger
import time

class VectorStore:
    def __init__(self, index_dir: str, model: str = "text-embedding-ada-002"):
        self.logger = RAGLogger('VectorStore')
        self.index_dir = Path(index_dir)
        self.model = model
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(1536) 

        self.logger.info(
            "VectorStore initialized",
            index_dir=index_dir,
            model=model,
            faiss_dimension=1536,
            index_type="IndexFlatL2"
        )

    def load_chunks(self, input_dir: str) -> List[str]:
        """Load text chunks from the input directory."""

        start_time = time.time()

        self.logger.info(
            "Starting chunk loading",
            input_directory = input_dir
        )

        input_path = Path(input_dir)
        chunks = []
        
        chunk_files_found = 0
        for file in input_path.iterdir():
            self.logger.debug(
                "Examining file",
                filename=file.name,
                is_chunk_file=file.name.endswith('_chunks.txt')
            )

            if file.name.endswith('_chunks.txt'):  # Fixed: check filename ending, not suffix
                chunk_files_found += 1
                
                self.logger.debug(
                    "Processing chunk file",
                    filename=file.name,
                    file_number=chunk_files_found
                )

                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                    if content:  # Only add non-empty chunks
                        # Split by double newlines (chunk separator)
                        file_chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                        self.logger.debug(
                            "Chunks extracted from file",
                            filename=file.name,
                            chunks_in_file=len(file_chunks),
                            file_size_chars=len(content)
                        )
                        chunks.extend(file_chunks)
                    else:
                        self.logger.warning(
                            "Empty chunk file encountered",
                            filename=file.name
                        )
        loading_time = time.time() - start_time

        self.logger.info(
        "Chunk loading completed",
        chunk_files_processed=chunk_files_found,
        total_chunks_loaded=len(chunks),
        loading_time_seconds=round(loading_time, 3),
        average_chunks_per_file=round(len(chunks) / chunk_files_found, 1) if chunk_files_found else 0
        )

        return chunks

    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks using Azure OpenAI API."""
        start_time = time.time()


        if not chunks:
            self.logger.warning("No chunks provided for embedding generation")
            return np.empty((0, 1536))
        
        self.logger.info(
            "Starting embedding generation",
            chunks_count=len(chunks),
            model=self.model,
            estimated_tokens=sum(len(chunk.split()) for chunk in chunks) * 1.3  # Rough estimate
         )
        
        try:
            azure_endpoint = settings.AZURE_OPENAI_ENDPOINT
            api_version = settings.AZURE_OPENAI_API_VERSION
            api_key = settings.AZURE_OPENAI_API_KEY
            
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            )

            self.logger.debug("Azure OpenAI client initialized successfully")


        except Exception as e:
            self.logger.error(
                "Failed to initialize Azure OpenAI client",
                error_type = type(e).__name__,
                error_message=str(e)
            )
            raise
        
        embeddings = []
        successful_embeddings = 0
        failed_embeddings = 0
        total_api_time = 0
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()

            self.logger.debug(
            "Processing chunk for embedding",
            chunk_number=i+1,
            total_chunks=len(chunks),
            chunk_length=len(chunk),
            chunk_words=len(chunk.split())
        )
            try:
                embedding = client.embeddings.create(input=chunk, model=settings.AZURE_EMBED_DEPLOYMENT).data[0].embedding
                embeddings.append(embedding)
                successful_embeddings += 1

                chunk_time = time.time() - chunk_start_time
                total_api_time += chunk_time

                self.logger.debug(
                    "Embedding generated successfully",
                    chunk_number=i+1,
                    embedding_dimension=len(embedding),
                    api_call_time_seconds=round(chunk_time, 3)
               )
            except Exception as e:

                failed_embeddings += 1
                chunk_time = time.time() - chunk_start_time
                total_api_time += chunk_time

                self.logger.error(
                    "Failed to generate embedding",
                    chunk_number=i+1,
                    chunk_length=len(chunk),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    api_call_time_seconds=round(chunk_time, 3)
                )
                raise
        
        result = np.vstack(embeddings)
        total_time = time.time() - start_time

        # Cost estimation (rough)
        estimated_tokens = sum(len(chunk.split()) for chunk in chunks) * 1.3
        estimated_cost = (estimated_tokens / 1000) * 0.0001  # Rough estimate
        
        self.logger.info(
            "Embedding generation completed",
            embeddings_shape=list(result.shape),
            successful_embeddings=successful_embeddings,
            failed_embeddings=failed_embeddings,
            total_api_calls=len(chunks),
            total_processing_time_seconds=round(total_time, 3),
            total_api_time_seconds=round(total_api_time, 3),
            average_time_per_embedding=round(total_api_time / len(chunks), 3),
            estimated_tokens=int(estimated_tokens),
            estimated_cost_usd=round(estimated_cost, 4)
        )
        return result

    def save_index(self):
        """Save FAISS index to disk."""
        start_time = time.time()
        index_path = self.index_dir / "faiss_index"
        
        self.logger.info(
            "Starting FAISS index save",
            index_path=str(index_path),
            vectors_count=self.index.ntotal,
            index_dimension=self.index.d
        )
        
        try:
            faiss.write_index(self.index, str(index_path))
            save_time = time.time() - start_time

            # verify file was created

            if index_path.exists():
                file_size = index_path.stat().st_size
                self.logger.info(
                    "FAISS index saved successfully",
                    index_file=str(index_path),
                    file_size_bytes=file_size,
                    file_size_mb=round(file_size / (1024*1024), 2),
                    vectors_saved=self.index.ntotal,
                    save_time_seconds=round(save_time, 3)
                )
            else:
                self.logger.error("Index file verification failed after save")

        except Exception as e:
            save_time = time.time() - start_time
        
            self.logger.error(
                "Failed to save FAISS index",
                index_path=str(index_path),
                vectors_count=self.index.ntotal,
                error_type=type(e).__name__,
                error_message=str(e),
                save_time_seconds=round(save_time, 3)
            )
            raise

    def load_index(self):
        """Load FAISS index from disk."""
        start_time = time.time()
        index_path = self.index_dir / "faiss_index"

        self.logger.debug(
            "Attempting to load FAISS index",
            index_path=str(index_path),
            file_exists=index_path.exists()
        )
        
        if index_path.exists():
            try:
                file_size = index_path.stat().st_size
                self.index = faiss.read_index(str(index_path))
                load_time = time.time() - start_time
                self.logger.info(
                    "FAISS index loaded successfully",
                    index_file=str(index_path),
                    file_size_mb=round(file_size / (1024*1024), 2),
                    vectors_loaded=self.index.ntotal,
                    index_dimension=self.index.d,
                    load_time_seconds=round(load_time, 3)
                )

            except Exception as e:
                load_time = time.time() - start_time
            
                self.logger.error(
                    "Failed to load FAISS index",
                    index_path=str(index_path),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    load_time_seconds=round(load_time, 3)
                )
                raise
        else:
            self.logger.info(
                "No existing FAISS index found, will create new one",
                index_path=str(index_path)
            )

    def build_index(self, input_dir: str):
        """Build FAISS index from text chunks."""
        start_time = time.time()
        self.logger.info(
            "Starting FAISS index building process",
            input_directory=input_dir,
            index_directory=str(self.index_dir)
        )

        chunks = self.load_chunks(input_dir)
        
        if not chunks:
            self.logger.warning(
                "No chunks found for index building",
                input_directory=input_dir
            )
            return
        
        embeddings = self.generate_embeddings(chunks)
        
        if embeddings.size == 0:
            self.logger.error("No embeddings generated, cannot build index")
            return
        
        # Add embeddings to FAISS index
        embedding_start_time = time.time()

        self.logger.info(
            "Adding embeddings to FAISS index",
            embeddings_shape=list(embeddings.shape),
            index_type=type(self.index).__name__
        )

        self.index.add(embeddings)

        embedding_add_time = time.time() - embedding_start_time
    
        self.logger.info(
            "Embeddings added to index successfully",
            vectors_in_index=self.index.ntotal,
            add_time_seconds=round(embedding_add_time, 3)
        )
        
        self.save_index()

        total_time = time.time() - start_time
    
        self.logger.info(
            "FAISS index building completed",
            total_chunks_processed=len(chunks),
            total_vectors_indexed=self.index.ntotal,
            index_dimension=self.index.d,
            total_build_time_seconds=round(total_time, 3),
            average_time_per_chunk=round(total_time / len(chunks), 3)
        )

if __name__ == "__main__":
    logger = RAGLogger('VectorStore-Main')
    logger.info("Starting vector store script execution")
    
    input_dir = "data/index"
    index_dir = "data/index"
    
    logger.info(
        "Script configuration",
        input_directory=input_dir,
        index_directory=index_dir
    )
    
    store = VectorStore(index_dir)
    store.build_index(input_dir)
    
    logger.info("Vector store script execution completed")
