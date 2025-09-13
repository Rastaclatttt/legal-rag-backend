from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal RAG Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request/Response models
class RAGRequest(BaseModel):
    query: str
    jurisdictions: List[str] = ["federal", "nova_scotia"]
    top_k: int = 5
    include_response: bool = True

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    jurisdiction: str
    chunk_id: str

class RAGResponse(BaseModel):
    query: str
    results: List[SearchResult]
    ai_response: Optional[str] = None
    total_results: int
    jurisdictions_searched: List[str]

# Global variables for storing vector data
vector_db = {}
embeddings_cache = {}

def load_embeddings_data():
    """Load embeddings data from JSON files"""
    global vector_db, embeddings_cache
    
    try:
        # Load federal data
        federal_path = "../legal_documents_with_embeddings.json"
        if os.path.exists(federal_path):
            logger.info("Loading federal embeddings...")
            with open(federal_path, 'r') as f:
                federal_data = json.load(f)
                vector_db['federal'] = federal_data
                logger.info(f"Loaded {len(federal_data)} federal documents")
        
        # Load Nova Scotia data  
        ns_path = "../NS_Consolidated_Statutes/ns_statutes_embeddings.json"
        if os.path.exists(ns_path):
            logger.info("Loading Nova Scotia embeddings...")
            with open(ns_path, 'r') as f:
                ns_data = json.load(f)
                vector_db['nova_scotia'] = ns_data
                logger.info(f"Loaded {len(ns_data)} Nova Scotia documents")
                
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        # Create empty databases if files don't exist
        vector_db['federal'] = []
        vector_db['nova_scotia'] = []

def get_embedding(text: str) -> List[float]:
    """Generate embedding for query text"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_documents(query_embedding: List[float], jurisdictions: List[str], top_k: int) -> List[SearchResult]:
    """Search for similar documents using cosine similarity"""
    results = []
    
    for jurisdiction in jurisdictions:
        if jurisdiction not in vector_db:
            continue
            
        jurisdiction_data = vector_db[jurisdiction]
        for doc in jurisdiction_data:
            if 'embedding' not in doc:
                continue
                
            similarity = cosine_similarity(query_embedding, doc['embedding'])
            
            result = SearchResult(
                content=doc.get('content', ''),
                metadata=doc.get('metadata', {}),
                similarity_score=similarity,
                jurisdiction=jurisdiction,
                chunk_id=doc.get('id', '')
            )
            results.append(result)
    
    # Sort by similarity score and return top_k
    results.sort(key=lambda x: x.similarity_score, reverse=True)
    return results[:top_k]

def generate_ai_response(query: str, search_results: List[SearchResult]) -> str:
    """Generate AI response based on search results"""
    try:
        # Prepare context from search results
        context = "\n\n".join([
            f"Source: {result.metadata.get('title', 'Unknown')} "
            f"({result.jurisdiction})\n{result.content}"
            for result in search_results[:3]  # Use top 3 results for context
        ])
        
        messages = [
            {
                "role": "system",
                "content": """You are a legal research assistant specializing in Canadian federal law and Nova Scotia provincial law. 
                Provide accurate, helpful responses based on the provided legal documents. Always cite your sources and 
                include appropriate legal disclaimers. If you cannot find relevant information in the provided context, 
                say so clearly."""
            },
            {
                "role": "user", 
                "content": f"""Based on the following legal documents, please answer this question: {query}

Legal Context:
{context}

Please provide a comprehensive answer citing the relevant laws and sections."""
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

@app.on_event("startup")
async def startup_event():
    """Load embeddings data on startup"""
    logger.info("Starting up Legal RAG Backend...")
    load_embeddings_data()
    logger.info("Startup complete!")

@app.get("/")
async def root():
    return {"message": "Legal RAG Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "federal_docs": len(vector_db.get('federal', [])),
        "nova_scotia_docs": len(vector_db.get('nova_scotia', []))
    }

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    """Main RAG endpoint for legal document search and AI response"""
    try:
        # Generate embedding for query
        query_embedding = get_embedding(request.query)
        
        # Search for similar documents
        search_results = search_similar_documents(
            query_embedding, 
            request.jurisdictions, 
            request.top_k
        )
        
        # Generate AI response if requested
        ai_response = None
        if request.include_response and search_results:
            ai_response = generate_ai_response(request.query, search_results)
        
        return RAGResponse(
            query=request.query,
            results=search_results,
            ai_response=ai_response,
            total_results=len(search_results),
            jurisdictions_searched=request.jurisdictions
        )
        
    except Exception as e:
        logger.error(f"Error in RAG endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))