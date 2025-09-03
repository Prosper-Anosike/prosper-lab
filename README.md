# Advancement RAG

Retrieval-Augmented Generation (RAG) pipeline for the **Advancement Project**.  
This project builds a chatbot that answers natural language questions based on documents stored in SharePoint.  

The system retrieves relevant excerpts from Advancement workstream documents, feeds them into a Large Language Model (LLM), and generates grounded answers with citations back to the original sources.

---

## 🎯 Why this project?
The Advancement Project involves multiple workstreams (*Prepare, Present, Place, Prove, Plan*) with large sets of documents in SharePoint.  
Manually searching through these files is slow and inconsistent.  

This chatbot will:
- Provide fast, accurate answers about Advancement workstreams.  
- Ensure responses are traceable with citations to the source docs.  
- Reduce time spent digging through SharePoint manually.  
- Create a foundation we can later scale into Microsoft’s Azure stack.  

---

## 🏗️ What we are building
1. **Document ingestion**: parse, chunk, and embed SharePoint documents.  
2. **Vector database**: store embeddings for fast semantic search.  
3. **Retriever**: fetch top-k relevant chunks for a query.  
4. **LLM**: generate an answer using retrieved context (with citations).  
5. **API**: simple endpoints (`/ingest`, `/ask`, `/health`, `/stats`) so apps or Teams integrations can call it.  

---

## ⚙️ Tech stack
- **Python 3.11**  
- **LlamaIndex** – orchestration of ingestion + retrieval  
- **FAISS / Chroma** – local vector store  
- **OpenAI GPT-4o-mini / Claude Sonnet** – chat model  
- **OpenAI text-embedding-3-large** – embedding model  
- **FastAPI** – API server  
- **pypdf, python-docx, python-pptx, openpyxl** – document parsing  
- **tiktoken** – chunking & token counting  

---

## 🔮 Roadmap
- ✅ Build local RAG pipeline (prototype).  
- ⏳ Add evaluation set and measure accuracy/groundedness.  
- ⏳ Connect directly to SharePoint for live ingestion.  
- ⏳ Migrate to Azure AI Search + AI Foundry for production.  
- ⏳ Add Teams/Planner integration for end-user access.  

## 📌 Status
Currently in **Phase 1 (local prototype)**:  
- Repo skeleton and architecture defined.  
- First ingestion and retrieval pipeline being implemented.  
- Running on exported SharePoint documents until direct access is granted.
