#!/usr/bin/env python3
"""
Legal RAG Backend API
FastAPI service for Vercel deployment with Qdrant integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
from qdrant_rag_service import LegalRAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="Legal Research Assistant Backend with Qdrant Vector Search",
    version="1.0.0"
)

# Configure CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://*.vercel.app",  # All Vercel subdomains
        "https://vercel.app",  # Vercel domain
        "*"  # Allow all origins for now - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
try:
    logger.info("🔧 Starting RAG service initialization...")
    rag_service = LegalRAGService()
    logger.info("✅ RAG service initialized successfully")

    # Debug logging
    if rag_service:
        logger.info(f"🔧 Qdrant client exists: {rag_service.qdrant_client is not None}")
        logger.info(f"🔧 OpenAI client exists: {rag_service.openai_client is not None}")
        if rag_service.openai_client:
            logger.info(f"🔧 OpenAI client type: {type(rag_service.openai_client)}")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG service: {e}")
    import traceback
    logger.error(f"❌ Traceback: {traceback.format_exc()}")
    rag_service = None

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    jurisdictions: Optional[List[str]] = ["federal", "nova_scotia"]
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    jurisdiction: str
    chunk_id: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    jurisdictions_searched: List[str]

class RAGRequest(BaseModel):
    query: str
    jurisdictions: Optional[List[str]] = ["federal", "nova_scotia"]
    top_k: Optional[int] = 5
    include_response: Optional[bool] = True

class RAGResponse(BaseModel):
    query: str
    results: List[SearchResult]
    ai_response: Optional[str] = None
    total_results: int
    jurisdictions_searched: List[str]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")

    return {
        "status": "healthy",
        "service": "Legal RAG API",
        "version": "1.0.0",
        "qdrant_connected": rag_service.is_connected()
    }

# Debug endpoint
@app.get("/debug")
async def debug_info():
    """Debug endpoint to see what's happening with initialization."""
    debug_info = {
        "rag_service_exists": rag_service is not None,
        "qdrant_client_exists": False,
        "openai_client_exists": False,
        "is_connected": False
    }

    if rag_service is not None:
        debug_info["qdrant_client_exists"] = rag_service.qdrant_client is not None
        debug_info["openai_client_exists"] = rag_service.openai_client is not None
        debug_info["is_connected"] = rag_service.is_connected()

        # Try to get more info about the clients
        if rag_service.qdrant_client:
            debug_info["qdrant_client_type"] = str(type(rag_service.qdrant_client))

        if rag_service.openai_client:
            debug_info["openai_client_type"] = str(type(rag_service.openai_client))

    return debug_info

# Search endpoint (vector search only)
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Perform vector search across legal documents."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        logger.info(f"Search request: {request.query[:50]}...")
        
        results = await rag_service.search(
            query=request.query,
            jurisdictions=request.jurisdictions,
            top_k=request.top_k
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            jurisdictions_searched=request.jurisdictions
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG endpoint (search + AI response)
@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """Perform full RAG query with AI-generated response."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        logger.info(f"RAG request: {request.query[:50]}...")
        
        # Perform search
        results = await rag_service.search(
            query=request.query,
            jurisdictions=request.jurisdictions,
            top_k=request.top_k
        )
        
        # Generate AI response if requested
        ai_response = None
        if request.include_response and results:
            ai_response = await rag_service.generate_response(
                query=request.query,
                search_results=results
            )
        
        return RAGResponse(
            query=request.query,
            results=results,
            ai_response=ai_response,
            total_results=len(results),
            jurisdictions_searched=request.jurisdictions
        )
        
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get available jurisdictions
@app.get("/jurisdictions")
async def get_jurisdictions():
    """Get list of available jurisdictions."""
    return {
        "jurisdictions": [
            {
                "id": "federal",
                "name": "Federal Canadian",
                "description": "Federal statutes and regulations"
            },
            {
                "id": "nova_scotia", 
                "name": "Nova Scotia",
                "description": "Nova Scotia provincial statutes"
            }
        ]
    }

# Get database statistics
@app.get("/stats")
async def get_statistics():
    """Get database statistics."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        stats = await rag_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Legal RAG API",
        "version": "1.0.0",
        "description": "Backend API for Legal Research Assistant",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "rag": "/rag",
            "jurisdictions": "/jurisdictions",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

# For Vercel deployment
handler = app

if __name__ == "__main__":
    import uvicorn

    # Get port from environment (Railway sets PORT automatically)
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )