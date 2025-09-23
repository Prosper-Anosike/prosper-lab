from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from pydantic import BaseModel
from packages.rag.loaders.document_loader import DocumentLoader
from packages.rag.retrieval.retriever import Retriever
from packages.rag.prompting.llm_prompting import LLMPrompting
from packages.rag.chunking.text_chunker import TextChunker
from packages.rag.store.vector_store import VectorStore
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
    try:
        Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
        Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

        loader = DocumentLoader(RAW_DIR, RAW_DIR)
        loader.process_documents()

        chunker = TextChunker(RAW_DIR, CHUNKS_DIR)
        chunker.process_text_files()

        store = VectorStore(INDEX_DIR)
        store.build_index(CHUNKS_DIR)

        return {"status": "success", "message": "Documents ingested successfully."}
    except Exception as e:
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
