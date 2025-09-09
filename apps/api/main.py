from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from packages.rag.loaders.document_loader import DocumentLoader
from packages.rag.retrieval.retriever import Retriever
from packages.rag.prompting.llm_prompting import LLMPrompting
from packages.rag.chunking.text_chunker import TextChunker
from packages.rag.store.vector_store import VectorStore

app = FastAPI()

class RequestQuery(BaseModel):
    query : str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/ingest")
def ingest_documents():
    """Trigger document ingestion."""
    try:
        input_dir = "data/raw"
        output_dir = "data/raw"
        index_dir = "data/index"

        # Step 1: Extract text from documents
        loader = DocumentLoader(input_dir, output_dir)
        loader.process_documents()
        print("done DocumentLoader")
        print("begin chunkking")
        # Step 2: Chunk the text
        chunker = TextChunker(input_dir,output_dir)
        chunker.process_text_files()
        print("done chunkking")
        print("begin vectorstore process")
        # Step 3: Generate embeddings and build index
        store = VectorStore(index_dir)
        store.build_index(input_dir )
        print("done vectorstore process")
        return {"status": "success", "message": "Documents ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: RequestQuery):
    """Accept natural language queries and return answers."""
    try:
        index_dir = "data/index"
        chunk_dir = "data/index"
        retriever = Retriever(index_dir)
        chunks = retriever.orchestrate_retrieval(request.query, chunk_dir)

        llm = LLMPrompting()    
        prompt = llm.generate_prompt(request.query, chunks)
        response = llm.get_response(prompt)

        return {"query": request.query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def system_stats():
    """Show system statistics."""
    try:
        raw_dir = Path("data/raw")
        index_dir = Path("data/index")

        raw_files = len(list(raw_dir.iterdir()))
        indexed_files = len(list(index_dir.iterdir()))

        return {
            "raw_files": raw_files,
            "indexed_files": indexed_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
