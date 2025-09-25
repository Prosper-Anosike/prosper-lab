from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from pydantic import BaseModel
from packages.rag.loaders.document_loader import DocumentLoader
from packages.rag.retrieval.retriever import Retriever
from packages.rag.prompting.llm_prompting import LLMPrompting
from packages.rag.chunking.text_chunker import TextChunker
from packages.rag.store.vector_store import VectorStore
from packages.rag.pipeline.ingestion_pipeline import IngestionPipeline
from utils.RAGLogger import RAGLogger
import time

app = FastAPI()

# Logger initialization
logger = RAGLogger('api')

RAW_DIR = "data/raw"
CHUNKS_DIR = "data/chunks"
INDEX_DIR = "data/index"

# Add middleware for logger

@app.middleware("http")
async def log_request(request: Request, call_next):
    start_time = time.time()

    logger.info(
        f"Request received: {request.method} {request.url.path}",
        method=request.method,
        path=request.url.path,
        client_ip=str(request.client.host)
    )

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {response.status_code}",
        status_code=response.status_code,
        process_time_ms = round(process_time * 1000, 2)
    )
    return response


class RequestQuery(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ingest")
def ingest_documents():
    """Legacy endpoint - use /ingestion for detailed results"""
    try:
        # Use the new improved pipeline
        pipeline = IngestionPipeline()
        results = pipeline.run_ingestion()
        
        # Build vector index after ingestion
        store = VectorStore(INDEX_DIR)
        store.build_index(INDEX_DIR)  # Use INDEX_DIR since chunks are now there
        
        # Return simple response for backward compatibility
        file_stats = results["file_statistics"]
        if file_stats["failed_files"] > 0:
            return {
                "status": "partial_success", 
                "message": f"Processed {file_stats['successful_files']}/{file_stats['total_files_discovered']} files successfully. {file_stats['failed_files']} files failed."
            }
        else:
            return {
                "status": "success", 
                "message": f"Successfully processed {file_stats['successful_files']} files and created {results['content_statistics']['total_chunks_created']} chunks."
            }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingestion")
def run_ingestion_pipeline():
    """
    Run the improved ingestion pipeline with detailed results.
    Returns comprehensive information about processing results.
    """
    try:
        logger.info("Starting ingestion pipeline via API")
        
        # Run the improved pipeline
        pipeline = IngestionPipeline()
        results = pipeline.run_ingestion()
        
        # Build vector index after ingestion
        logger.info("Building vector index")
        store = VectorStore(INDEX_DIR)
        store.build_index(INDEX_DIR)  # Use INDEX_DIR since chunks are now there
        
        # Add vector index info to results
        results["vector_index"] = {
            "status": "built",
            "index_path": INDEX_DIR,
            "chunks_indexed": results["content_statistics"]["total_chunks_created"]
        }
        
        logger.info(
            "Ingestion pipeline completed via API",
            files_processed=results["file_statistics"]["files_processed"],
            successful_files=results["file_statistics"]["successful_files"],
            total_chunks=results["content_statistics"]["total_chunks_created"]
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed via API: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/ingestion/status")
def get_ingestion_status():
    """Get current ingestion pipeline status"""
    try:
        pipeline = IngestionPipeline()
        status = pipeline.get_pipeline_status()
        
        # Add vector index status
        index_path = Path(INDEX_DIR) / "faiss_index"
        status["vector_index_exists"] = index_path.exists()
        
        return status
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: RequestQuery):
    try:
        retriever = Retriever(INDEX_DIR)
        chunks = retriever.orchestrate_retrieval(request.query, CHUNKS_DIR)

        # No-retrieval guard: don't let the LLM improvise
        if not chunks or (len(chunks) == 1 and chunks[0] == "[NO_RELEVANT_SOURCES_FOUND]"):
            return {
                "query": request.query,
                "response": "No relevant sources found.",
                "chunks_used": 0
            }

        llm = LLMPrompting()
        messages = llm.generate_prompt(request.query, chunks)
        response = llm.get_response(messages)

        return {"query": request.query, "response": response, "chunks_used": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def system_stats():
    try:
        raw_dir = Path(RAW_DIR)
        chunks_dir = Path(CHUNKS_DIR)
        index_dir = Path(INDEX_DIR)

        raw_files = len([p for p in raw_dir.iterdir() if p.is_file()])
        chunk_files = len([p for p in chunks_dir.iterdir() if p.is_file() and p.name.endswith("_chunks.txt")])
        has_index = (index_dir / "faiss_index").exists()

        return {
            "raw_files": raw_files,
            "chunk_files": chunk_files,
            "index_exists": has_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
