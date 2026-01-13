import json
import hashlib
import faiss
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
from sentence_transformers import SentenceTransformer

from configs.settings import settings
from utils.RAGLogger import RAGLogger

LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_dir: str, top_k: int = 5):
        self.logger = RAGLogger('Retriever')
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_index"
        self.mapping_path = self.index_dir / "chunks.json"
        self.top_k = top_k
        requested_model = (settings.EMBED_MODEL or LOCAL_EMBED_MODEL).strip()
        normalized_model = requested_model.lower()
        remote_tokens = ("text-embedding", "gpt", "openai")
        looks_remote = any(token in normalized_model for token in remote_tokens)
        unresolved_path = "/" in requested_model and not Path(requested_model).exists()

        if looks_remote or unresolved_path:
            self.logger.warning(
                "Remote or unavailable embedding model requested; falling back to local sentence-transformer",
                requested_model=requested_model,
                fallback_model=LOCAL_EMBED_MODEL,
                reason="remote-indicator" if looks_remote else "missing-local-path"
            )
            self.model = LOCAL_EMBED_MODEL
        else:
            self.model = requested_model
        self.embedder = SentenceTransformer(self.model)
        self.embedding_dimension = self.embedder.get_sentence_embedding_dimension()
        
        self.logger.info(
            "Retriever initialized",
            index_dir=index_dir,
            top_k=top_k,
            model=self.model,
            embedding_dimension=self.embedding_dimension,
            index_path=str(self.index_path),
            mapping_path=str(self.mapping_path)
        )
        
        self.index = self.load_index()
        self._chunk_lookup_dir: str | None = None
        self._chunk_lookup: Dict[str, Dict[str, Any]] = {}

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
            encode_start = time.time()
            embedding = self.embedder.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            encode_time = time.time() - encode_start

            result = np.array([embedding], dtype="float32")
            total_time = time.time() - start_time
            
            self.logger.info(
                "Query embedding generated successfully",
                embedding_shape=list(result.shape),
                embedding_dimension=len(embedding),
                embedding_generation_time_seconds=round(encode_time, 3),
                total_time_seconds=round(total_time, 3)
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

    def retrieve(self, query_embedding: np.ndarray) -> Tuple[List[int], List[float]]:
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
            
            return indices_list, distance_list
            
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

    def _build_preview(self, text: str, limit: int = 160) -> str:
        snippet = " ".join(text.split())
        return snippet if len(snippet) <= limit else snippet[:limit].rstrip() + "..."

    def _ensure_chunk_lookup(self, chunk_dir: str) -> Dict[str, Dict[str, Any]]:
        if self._chunk_lookup_dir == chunk_dir and self._chunk_lookup:
            return self._chunk_lookup

        lookup: Dict[str, Dict[str, Any]] = {}
        chunk_path = Path(chunk_dir)
        if not chunk_path.exists():
            self.logger.warning(
                "Chunk directory missing for metadata lookup",
                chunk_directory=chunk_dir
            )
            self._chunk_lookup_dir = chunk_dir
            self._chunk_lookup = lookup
            return lookup

        total_chunks = 0
        for file in sorted(chunk_path.iterdir(), key=lambda p: p.name):
            if not file.name.endswith("_chunks.txt"):
                continue

            doc_name = file.name.replace("_chunks.txt", "")
            try:
                content = file.read_text(encoding="utf-8").strip()
            except Exception as exc:
                self.logger.error(
                    "Failed to read chunk file for metadata lookup",
                    filename=file.name,
                    error_type=type(exc).__name__,
                    error_message=str(exc)
                )
                continue

            if not content:
                continue

            chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
            for idx, chunk_text in enumerate(chunks, start=1):
                digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
                lookup[digest] = {
                    "doc_name": doc_name,
                    "chunk_index": idx,
                    "source_path": str(file),
                    "preview": self._build_preview(chunk_text),
                }
                total_chunks += 1

        self.logger.info(
            "Chunk metadata cache refreshed",
            chunk_directory=chunk_dir,
            available_entries=len(lookup),
            total_chunks_scanned=total_chunks
        )

        self._chunk_lookup_dir = chunk_dir
        self._chunk_lookup = lookup
        return lookup

    def _resolve_metadata(self, chunk_text: str, chunk_dir: str, fallback_index: int) -> Dict[str, Any]:
        lookup = self._ensure_chunk_lookup(chunk_dir)
        digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
        metadata = lookup.get(digest)
        if metadata:
            return metadata

        return {
            "doc_name": "unknown-source",
            "chunk_index": fallback_index + 1,
            "source_path": "",
            "preview": self._build_preview(chunk_text)
        }

    def _build_citations(self, chunk_entries: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for rank, entry in enumerate(chunk_entries, start=1):
            metadata = entry.get("metadata", {})
            doc_name = metadata.get("doc_name", "unknown-source")
            chunk_index = metadata.get("chunk_index", rank)
            source_path = metadata.get("source_path", "")
            if source_path:
                source_path = Path(source_path).as_posix()
            score = scores[rank - 1] if scores and len(scores) >= rank else None

            citations.append(
                {
                    "id": f"S{rank}",
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "source_path": source_path,
                    "preview": metadata.get("preview", ""),
                    "score": score,
                    "label": f"{doc_name} · chunk {chunk_index}",
                }
            )

        return citations

    def fetch_chunks(self, indices: List[int], chunk_dir: str) -> List[Dict[str, Any]]:
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
            chunk_entries: List[Dict[str, Any]] = []
            successful_fetches = 0
            failed_fetches = 0
            total_characters = 0
            
            for idx in indices:
                if 0 <= idx < len(all_chunks):
                    raw_entry = all_chunks[idx]

                    if isinstance(raw_entry, dict) and "text" in raw_entry:
                        chunk_content = raw_entry["text"]
                        metadata = {
                            "doc_name": raw_entry.get("doc_name", "unknown-source"),
                            "chunk_index": raw_entry.get("chunk_index", idx + 1),
                            "source_path": raw_entry.get("source_path", ""),
                            "preview": raw_entry.get("preview", self._build_preview(raw_entry.get("text", "")))
                        }
                    else:
                        chunk_content = str(raw_entry)
                        metadata = self._resolve_metadata(chunk_content, chunk_dir, idx)

                    chunk_entries.append({
                        "text": chunk_content,
                        "metadata": metadata
                    })
                    successful_fetches += 1
                    total_characters += len(chunk_content)
                    
                    self.logger.debug(
                        "Chunk fetched successfully",
                        chunk_index=idx,
                        chunk_length=len(chunk_content),
                        chunk_words=len(chunk_content.split()),
                        chunk_preview=metadata.get("preview")
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
            
            return chunk_entries
            
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

    def orchestrate_retrieval(self, query: str, chunk_dir: str) -> Dict[str, Any]:
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
            indices, scores = self.retrieve(query_embedding)
            search_time = time.time() - search_start_time
            
            # Step 3: Fetch actual chunks
            fetch_start_time = time.time()
            chunk_entries = self.fetch_chunks(indices, chunk_dir)
            fetch_time = time.time() - fetch_start_time
            
            total_time = time.time() - start_time
            
            chunk_texts = [entry["text"] for entry in chunk_entries]
            citations = self._build_citations(chunk_entries, scores)
            total_characters = sum(len(chunk) for chunk in chunk_texts)
            avg_chunk_length = round(total_characters / len(chunk_texts), 1) if chunk_texts else 0
            
            self.logger.info(
                "Retrieval orchestration completed successfully",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                chunks_retrieved=len(chunk_texts),
                total_characters_retrieved=total_characters,
                average_chunk_length=avg_chunk_length,
                embedding_time_seconds=round(embedding_time, 3),
                search_time_seconds=round(search_time, 3),
                fetch_time_seconds=round(fetch_time, 3),
                total_time_seconds=round(total_time, 3),
                retrieval_efficiency=round((len(chunk_texts) / self.top_k) * 100, 1) if self.top_k else 0,
                citations=len(citations)
            )
            
            return {
                "chunks": chunk_texts,
                "citations": citations,
                "scores": scores,
                "stats": {
                    "embedding_time_seconds": round(embedding_time, 3),
                    "search_time_seconds": round(search_time, 3),
                    "fetch_time_seconds": round(fetch_time, 3),
                    "total_time_seconds": round(total_time, 3),
                    "avg_chunk_length": avg_chunk_length,
                    "chunks_retrieved": len(chunk_texts),
                }
            }
            
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
