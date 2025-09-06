import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time

from backend.core.ingest import create_embeddings_and_store, load_and_chunk_document_from_string
from backend.core.query import initialize_retriever, retrieve_and_rerank, generate_answer_with_citations

app = FastAPI(title="Mini RAG App")

origins = [
    "http://localhost:5173",  
    "http://localhost:3000",  
    os.environ.get("CORS_ORIGIN") 
]

origins = [origin for origin in origins if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    retriever_instance = initialize_retriever()
except Exception as e:
    print(f"Failed to initialize retriever on startup: {e}")
    retriever_instance = None

class QueryRequest(BaseModel):
    query: str

class TextIngestRequest(BaseModel):
    text: str

@app.post("/ingest")
async def ingest_document(request: TextIngestRequest):
    try:
        start_time = time.time()
        chunks = load_and_chunk_document_from_string(request.text)
        create_embeddings_and_store(chunks)

        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "message": "Document ingested successfully!", 
            "chunks_processed": len(chunks),
            "processing_time": f"{elapsed_time:.2f}s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/query")
async def answer_query(request: QueryRequest):
    global retriever_instance 
    if not retriever_instance:
        try:
            retriever_instance = initialize_retriever()
        except Exception as e:
             raise HTTPException(status_code=503, detail=f"RAG system not ready. Failed to initialize retriever: {e}")

    user_query = request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        start_time = time.time()
        reranked_docs = retrieve_and_rerank(user_query, retriever_instance)
        result = generate_answer_with_citations(user_query, reranked_docs)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            "query": user_query,
            "answer": result["answer"],
            "sources": result["sources"],
            "processing_time": f"{elapsed_time:.2f}s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)