#!/usr/bin/env python3
"""
Legal RAG Service
Core service class for Pinecone vector search and OpenAI response generation
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

try:
    from pinecone import Pinecone
    import openai
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result from vector database."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    jurisdiction: str
    chunk_id: str

class LegalRAGService:
    """Service for legal document search and RAG operations."""
    
    def __init__(self):
        """Initialize the RAG service with Pinecone and OpenAI."""
        self.pinecone_client = None
        self.index = None
        self.openai_client = None
        
        self._initialize_pinecone()
        self._initialize_openai()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection."""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            index_name = os.getenv('PINECONE_INDEX_NAME', 'legal-rag-index')
            
            if not api_key:
                logger.error("PINECONE_API_KEY not found in environment")
                return
            
            self.pinecone_client = Pinecone(api_key=api_key)
            self.index = self.pinecone_client.Index(index_name)
            
            logger.info(f"✅ Pinecone initialized: {index_name}")
            
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
    
    def _initialize_openai(self):
        """Initialize OpenAI connection."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment")
                return
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("✅ OpenAI initialized")
            
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
    
    def is_connected(self) -> bool:
        """Check if service is properly connected."""
        return self.index is not None and self.openai_client is not None
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using OpenAI."""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query,
                    encoding_format="float"
                )
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        jurisdictions: List[str] = ["federal", "nova_scotia"],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for relevant legal documents."""
        if not self.index:
            raise Exception("Pinecone index not initialized")
        
        try:
            # Get query embedding
            query_embedding = await self.get_query_embedding(query)
            
            all_results = []
            
            # Search each jurisdiction namespace
            for jurisdiction in jurisdictions:
                namespace = jurisdiction
                
                # Perform vector search
                loop = asyncio.get_event_loop()
                search_response = await loop.run_in_executor(
                    None,
                    lambda: self.index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace
                    )
                )
                
                # Process results
                for match in search_response.matches:
                    result = SearchResult(
                        content=self._extract_content(match.metadata),
                        metadata=match.metadata,
                        similarity_score=float(match.score),
                        jurisdiction=jurisdiction,
                        chunk_id=match.id
                    )
                    all_results.append(result)
            
            # Sort by similarity score
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top results across all jurisdictions
            return all_results[:top_k * len(jurisdictions)]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def _extract_content(self, metadata: Dict[str, Any]) -> str:
        """Extract content from metadata."""
        # Try different possible content fields
        content_fields = ['content_preview', 'content', 'text', 'description']
        
        for field in content_fields:
            if field in metadata and metadata[field]:
                return str(metadata[field])
        
        # Fallback: create content from available metadata
        title = metadata.get('title', 'Unknown Document')
        section = metadata.get('section', '')
        doc_type = metadata.get('document_type', '')
        
        fallback_content = f"Document: {title}"
        if section:
            fallback_content += f", Section: {section}"
        if doc_type:
            fallback_content += f", Type: {doc_type}"
        
        return fallback_content
    
    async def generate_response(
        self, 
        query: str, 
        search_results: List[SearchResult],
        max_context_length: int = 8000
    ) -> str:
        """Generate AI response using search results."""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        if not search_results:
            return "No relevant legal documents found to answer your question."
        
        try:
            # Prepare context from search results
            context_parts = []
            current_length = 0
            
            for result in search_results:
                result_text = f"""
Jurisdiction: {result.jurisdiction.title()}
Document: {result.metadata.get('title', 'Unknown')}
Section: {result.metadata.get('section', 'N/A')}
Content: {result.content}
"""
                if current_length + len(result_text) < max_context_length:
                    context_parts.append(result_text)
                    current_length += len(result_text)
                else:
                    break
            
            context = "\n---\n".join(context_parts)
            
            system_prompt = """You are a legal research assistant specializing in Canadian law. You have access to both Federal Canadian statutes/regulations and Nova Scotia provincial statutes.

Guidelines:
1. Always cite the specific jurisdiction (Federal or Nova Scotia) when referencing laws
2. Provide accurate legal information based only on the provided context
3. If information spans multiple jurisdictions, clearly distinguish between them
4. Include relevant section numbers and document titles when available
5. If the query cannot be answered from the provided context, say so clearly
6. Add appropriate legal disclaimers when giving legal information

Format your response clearly with proper citations."""

            user_prompt = f"""Based on the following legal documents, please answer this question: {query}

Legal Context:
{context}

Please provide a comprehensive answer with proper citations to the relevant laws and sections."""

            # Generate response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
            )
            
            ai_response = response.choices[0].message.content
            
            # Add disclaimer
            disclaimer = "\n\n**Legal Disclaimer:** This information is for research purposes only and does not constitute legal advice. Always consult with a qualified legal professional for legal matters."
            
            return ai_response + disclaimer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating AI response: {e}\n\nPlease refer to the search results above."
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.index:
            raise Exception("Pinecone index not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: self.index.describe_index_stats()
            )
            
            # Process namespace stats
            namespaces = stats.get('namespaces', {})
            
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 1536),
                "namespaces": {
                    name: {
                        "vector_count": info.get('vector_count', 0)
                    }
                    for name, info in namespaces.items()
                },
                "index_fullness": stats.get('index_fullness', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise