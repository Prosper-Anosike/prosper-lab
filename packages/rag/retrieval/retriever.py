import json
import faiss
import numpy as np
from pathlib import Path
from typing import List
from openai import AzureOpenAI
from configs.settings import settings
from utils.RAGLogger import RAGLogger
import time

class Retriever:
    def __init__(self, index_dir: str, top_k: int = 5):
        self.logger = RAGLogger('Retriever')
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_index"
        self.mapping_path = self.index_dir / "chunks.json"
        self.top_k = top_k
        self.model = settings.AZURE_EMBED_DEPLOYMENT
        
        self.logger.info(
            "Retriever initialized",
            index_dir=index_dir,
            top_k=top_k,
            model=self.model,
            index_path=str(self.index_path),
            mapping_path=str(self.mapping_path)
        )
        
        self.index = self.load_index()

    def load_index(self):
        start_time = time.time()
        
        self.logger.debug(
            "Attempting to load FAISS index",
            index_path=str(self.index_path),
            file_exists=self.index_path.exists()
        )
        
        if self.index_path.exists():
            try:
                file_size = self.index_path.stat().st_size
                index = faiss.read_index(str(self.index_path))
                load_time = time.time() - start_time
                
                self.logger.info(
                    "FAISS index loaded successfully",
                    index_file=str(self.index_path),
                    file_size_mb=round(file_size / (1024*1024), 2),
                    vectors_loaded=index.ntotal,
                    index_dimension=index.d,
                    load_time_seconds=round(load_time, 3)
                )
                
                return index
                
            except Exception as e:
                load_time = time.time() - start_time
                
                self.logger.error(
                    "Failed to load FAISS index",
                    index_path=str(self.index_path),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    load_time_seconds=round(load_time, 3)
                )
                raise
        else:
            self.logger.error(
                "FAISS index file not found",
                index_path=str(self.index_path),
                directory_exists=self.index_dir.exists()
            )
            raise FileNotFoundError("FAISS index not found.")

    def generate_query_embedding(self, query: str) -> np.ndarray:
        start_time = time.time()
        
        self.logger.debug(
            "Starting query embedding generation",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            query_length=len(query),
            query_words=len(query.split()),
            model=self.model
        )
        
        try:
            client = AzureOpenAI(
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
            )
            
            self.logger.debug("Azure OpenAI client initialized successfully")
            
            api_start_time = time.time()
            embedding = client.embeddings.create(input=query, model=self.model).data[0].embedding
            api_time = time.time() - api_start_time
            
            result = np.array([embedding], dtype="float32")
            total_time = time.time() - start_time
            
            # Estimate cost (rough calculation)
            estimated_tokens = len(query.split()) * 1.3
            estimated_cost = (estimated_tokens / 1000) * 0.0001
            
            self.logger.info(
                "Query embedding generated successfully",
                embedding_shape=list(result.shape),
                embedding_dimension=len(embedding),
                api_call_time_seconds=round(api_time, 3),
                total_time_seconds=round(total_time, 3),
                estimated_tokens=int(estimated_tokens),
                estimated_cost_usd=round(estimated_cost, 6)
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.error(
                "Failed to generate query embedding",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                query_length=len(query),
                error_type=type(e).__name__,
                error_message=str(e),
                total_time_seconds=round(total_time, 3)
            )
            raise

    def retrieve(self, query_embedding: np.ndarray) -> List[int]:
        start_time = time.time()
        
        self.logger.debug(
            "Starting similarity search",
            top_k=self.top_k,
            query_embedding_shape=list(query_embedding.shape),
            index_vectors_count=self.index.ntotal,
            index_dimension=self.index.d
        )
        
        try:
            search_start_time = time.time()
            distances, indices = self.index.search(query_embedding, self.top_k)
            search_time = time.time() - search_start_time
            
            # Convert to lists for logging
            distance_list = distances[0].tolist()
            indices_list = indices.flatten().tolist()
            
            # Calculate search quality metrics
            min_distance = float(min(distance_list)) if distance_list else 0.0
            max_distance = float(max(distance_list)) if distance_list else 0.0
            avg_distance = float(sum(distance_list) / len(distance_list)) if distance_list else 0.0
            
            total_time = time.time() - start_time
            
            self.logger.info(
                "Similarity search completed",
                results_found=len(indices_list),
                search_time_ms=round(search_time * 1000, 1),
                total_time_ms=round(total_time * 1000, 1),
                similarity_scores=distance_list,
                retrieved_indices=indices_list,
                min_distance=round(min_distance, 4),
                max_distance=round(max_distance, 4),
                avg_distance=round(avg_distance, 4),
                search_quality="excellent" if avg_distance < 0.5 else "good" if avg_distance < 1.0 else "fair"
            )
            
            return indices_list
            
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.error(
                "Similarity search failed",
                top_k=self.top_k,
                query_embedding_shape=list(query_embedding.shape),
                index_vectors_count=self.index.ntotal,
                error_type=type(e).__name__,
                error_message=str(e),
                total_time_ms=round(total_time * 1000, 1)
            )
            raise

    def load_all_chunks(self, chunk_dir: str) -> List[str]:
        start_time = time.time()
        
        self.logger.debug(
            "Starting chunk loading from directory",
            chunk_directory=chunk_dir
        )
        
        chunk_path = Path(chunk_dir)
        all_chunks: List[str] = []
        chunk_files_processed = 0
        total_characters = 0
        
        if not chunk_path.exists():
            self.logger.warning(
                "Chunk directory does not exist",
                chunk_directory=chunk_dir
            )
            return all_chunks
        
        for file in sorted(chunk_path.iterdir(), key=lambda p: p.name):
            if file.name.endswith("_chunks.txt"):
                chunk_files_processed += 1
                
                self.logger.debug(
                    "Processing chunk file",
                    filename=file.name,
                    file_number=chunk_files_processed,
                    file_size_bytes=file.stat().st_size
                )
                
                try:
                    content = file.read_text(encoding="utf-8").strip()
                    if content:
                        file_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                        file_characters = sum(len(chunk) for chunk in file_chunks)
                        
                        self.logger.debug(
                            "Chunks extracted from file",
                            filename=file.name,
                            chunks_in_file=len(file_chunks),
                            file_characters=file_characters,
                            average_chunk_length=round(file_characters / len(file_chunks), 1) if file_chunks else 0
                        )
                        
                        all_chunks.extend(file_chunks)
                        total_characters += file_characters
                    else:
                        self.logger.warning(
                            "Empty chunk file encountered",
                            filename=file.name
                        )
                        
                except Exception as e:
                    self.logger.error(
                        "Failed to read chunk file",
                        filename=file.name,
                        error_type=type(e).__name__,
                        error_message=str(e)
                    )
        
        loading_time = time.time() - start_time
        
        self.logger.info(
            "Chunk loading completed",
            chunk_directory=chunk_dir,
            chunk_files_processed=chunk_files_processed,
            total_chunks_loaded=len(all_chunks),
            total_characters=total_characters,
            average_chunk_length=round(total_characters / len(all_chunks), 1) if all_chunks else 0,
            loading_time_seconds=round(loading_time, 3)
        )
        
        return all_chunks

    def fetch_chunks(self, indices: List[int], chunk_dir: str) -> List[str]:
        start_time = time.time()
        
        self.logger.debug(
            "Starting chunk fetching",
            requested_indices=indices,
            chunk_directory=chunk_dir,
            mapping_file_exists=self.mapping_path.exists()
        )
        
        try:
            # Load chunks from mapping file or directory scan
            if self.mapping_path.exists():
                self.logger.debug(
                    "Using mapping file for chunk retrieval",
                    mapping_file=str(self.mapping_path),
                    file_size_bytes=self.mapping_path.stat().st_size
                )
                
                load_start_time = time.time()
                all_chunks = json.loads(self.mapping_path.read_text(encoding="utf-8"))
                load_time = time.time() - load_start_time
                
                self.logger.debug(
                    "Mapping file loaded successfully",
                    total_chunks_available=len(all_chunks),
                    load_time_seconds=round(load_time, 3)
                )
            else:
                self.logger.info(
                    "Mapping file not found, falling back to directory scan",
                    mapping_file=str(self.mapping_path),
                    chunk_directory=chunk_dir
                )
                all_chunks = self.load_all_chunks(chunk_dir)
            
            # Fetch requested chunks
            chunks = []
            successful_fetches = 0
            failed_fetches = 0
            total_characters = 0
            
            for idx in indices:
                if 0 <= idx < len(all_chunks):
                    chunk_content = all_chunks[idx]
                    chunks.append(chunk_content)
                    successful_fetches += 1
                    total_characters += len(chunk_content)
                    
                    self.logger.debug(
                        "Chunk fetched successfully",
                        chunk_index=idx,
                        chunk_length=len(chunk_content),
                        chunk_words=len(chunk_content.split()),
                        chunk_preview=chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
                    )
                else:
                    failed_fetches += 1
                    self.logger.warning(
                        "Chunk index out of range",
                        requested_index=idx,
                        total_chunks_available=len(all_chunks),
                        max_valid_index=len(all_chunks) - 1 if all_chunks else -1
                    )
            
            fetch_time = time.time() - start_time
            
            self.logger.info(
                "Chunk fetching completed",
                requested_chunks=len(indices),
                successful_fetches=successful_fetches,
                failed_fetches=failed_fetches,
                success_rate=round((successful_fetches / len(indices)) * 100, 1) if indices else 0,
                total_characters_fetched=total_characters,
                average_chunk_length=round(total_characters / successful_fetches, 1) if successful_fetches else 0,
                fetch_time_seconds=round(fetch_time, 3)
            )
            
            return chunks
            
        except Exception as e:
            fetch_time = time.time() - start_time
            
            self.logger.error(
                "Failed to fetch chunks",
                requested_indices=indices,
                chunk_directory=chunk_dir,
                error_type=type(e).__name__,
                error_message=str(e),
                fetch_time_seconds=round(fetch_time, 3)
            )
            raise

    def orchestrate_retrieval(self, query: str, chunk_dir: str) -> List[str]:
        start_time = time.time()
        
        self.logger.info(
            "Starting retrieval orchestration",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            query_length=len(query),
            query_words=len(query.split()),
            chunk_directory=chunk_dir,
            top_k=self.top_k
        )
        
        try:
            # Step 1: Generate query embedding
            embedding_start_time = time.time()
            query_embedding = self.generate_query_embedding(query)
            embedding_time = time.time() - embedding_start_time
            
            # Step 2: Perform similarity search
            search_start_time = time.time()
            indices = self.retrieve(query_embedding)
            search_time = time.time() - search_start_time
            
            # Step 3: Fetch actual chunks
            fetch_start_time = time.time()
            chunks = self.fetch_chunks(indices, chunk_dir)
            fetch_time = time.time() - fetch_start_time
            
            total_time = time.time() - start_time
            
            # Calculate retrieval quality metrics
            total_characters = sum(len(chunk) for chunk in chunks)
            avg_chunk_length = round(total_characters / len(chunks), 1) if chunks else 0
            
            self.logger.info(
                "Retrieval orchestration completed successfully",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                chunks_retrieved=len(chunks),
                total_characters_retrieved=total_characters,
                average_chunk_length=avg_chunk_length,
                embedding_time_seconds=round(embedding_time, 3),
                search_time_seconds=round(search_time, 3),
                fetch_time_seconds=round(fetch_time, 3),
                total_time_seconds=round(total_time, 3),
                retrieval_efficiency=round((len(chunks) / self.top_k) * 100, 1) if self.top_k else 0
            )
            
            return chunks
            
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.error(
                "Retrieval orchestration failed",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                chunk_directory=chunk_dir,
                error_type=type(e).__name__,
                error_message=str(e),
                total_time_seconds=round(total_time, 3)
            )
            raise

if __name__ == "__main__":
    logger = RAGLogger('Retriever-Main')
    logger.info("Starting retriever script execution")
    
    index_dir = "data/index"
    chunk_dir = "data/chunks"
    query = "What is the process for document ingestion?"
    
    logger.info(
        "Script configuration",
        index_directory=index_dir,
        chunk_directory=chunk_dir,
        query=query
    )
    
    try:
        retriever = Retriever(index_dir)
        results = retriever.orchestrate_retrieval(query, chunk_dir)
        
        logger.info(
            "Retriever script execution completed successfully",
            results_count=len(results),
            total_characters=sum(len(chunk) for chunk in results)
        )
        
        # Print results for demonstration
        for i, chunk in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            
    except Exception as e:
        logger.error(
            "Retriever script execution failed",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise
