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
import asyncio
from qdrant_rag_service import LegalRAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="Legal Research Assistant Backend with Qdrant Vector Search",
    version="1.0.1"  # Force redeploy
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

# Test OpenAI directly in endpoint
@app.get("/test-openai")
async def test_openai_direct():
    """Test OpenAI client creation directly in endpoint."""
    import os
    try:
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            return {"error": "No API key found", "api_key_present": False}

        client = OpenAI()
        return {
            "success": True,
            "client_type": str(type(client)),
            "api_key_present": True,
            "has_embeddings": hasattr(client, 'embeddings'),
            "has_chat": hasattr(client, 'chat')
        }
    except Exception as e:
        return {"error": str(e), "exception_type": str(type(e))}

# Database population endpoint
@app.post("/populate-database")
async def populate_database():
    """Populate database with sample Nova Scotia statutes."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not available")

    if not rag_service.qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not available")

    try:
        from qdrant_client.http.models import Distance, VectorParams, PointStruct
        import uuid

        # Sample Nova Scotia statute data (embedded directly to avoid file issues)
        sample_data = [
            {
                "text": "CHAPTER 9 OF THE REVISED STATUTES, 1989 - AIDS Advisory Commission Act. Short title: This Act may be cited as the AIDS Advisory Commission Act. Purpose: The purpose of this Act is to appoint an advisory body to advise the Government of Nova Scotia on issues related to the acquired immune deficiency syndrome.",
                "embedding": [0.047, -0.008, 0.029, 0.044] + [0.0] * 1532  # Dummy embedding for testing
            },
            {
                "text": "Accessibility Act - An Act Respecting Accessibility Standards. The purpose of this Act is to establish accessibility standards to address barriers that prevent persons with disabilities from participating fully in society.",
                "embedding": [0.023, -0.015, 0.041, 0.012] + [0.0] * 1532  # Dummy embedding for testing
            }
        ]

        collection_name = "nova_scotia_statutes"

        # Create collection if it doesn't exist
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: rag_service.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
            )
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection {collection_name} already exists")
            else:
                raise e

        # Create points
        points = []
        for i, item in enumerate(sample_data):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=item['embedding'],
                payload={
                    'content': item['text'],
                    'jurisdiction': 'nova_scotia',
                    'document_type': 'statute',
                    'title': f'Sample Statute {i+1}'
                }
            ))

        # Upload points
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: rag_service.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
        )

        # Get collection info
        collection_info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: rag_service.qdrant_client.get_collection(collection_name)
        )

        return {
            "success": True,
            "message": f"Imported {len(points)} sample statutes",
            "collection": collection_name,
            "total_points": collection_info.points_count
        }

    except Exception as e:
        logger.error(f"Database population error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reinit endpoint to fix OpenAI client
@app.get("/reinit-openai")
async def reinit_openai():
    """Reinitialize OpenAI client."""
    global rag_service
    if rag_service is None:
        return {"error": "RAG service not available"}

    try:
        # Re-initialize OpenAI client
        rag_service._initialize_openai()

        return {
            "success": True,
            "openai_client_exists": rag_service.openai_client is not None,
            "client_type": str(type(rag_service.openai_client)) if rag_service.openai_client else None,
            "is_connected": rag_service.is_connected()
        }
    except Exception as e:
        return {"error": str(e), "exception_type": str(type(e))}

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