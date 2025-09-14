#!/usr/bin/env python3
"""
Legal RAG Service with Qdrant
Core service class for Qdrant vector search and OpenAI response generation
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError as e:
    print(f"Qdrant import error: {e}")
    QDRANT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"OpenAI import error: {e}")
    OPENAI_AVAILABLE = False
    OpenAI = None

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
    """Service for legal document search and RAG operations with Qdrant."""

    def __init__(self):
        """Initialize the RAG service with Qdrant and OpenAI."""
        self.qdrant_client = None
        self.openai_client = None

        # Collection mapping for different jurisdictions
        self.collection_mapping = {
            "federal": [
                "federal_statutes",
                "federal_regulations",
                "federal_acts_a_c",
                "federal_acts_d_h",
                "federal_acts_i_m",
                "federal_acts_n_s",
                "federal_acts_t_z",
                "federal_other"
            ],
            "nova_scotia": [
                "ns_administrative",
                "ns_business_commerce",
                "ns_criminal_justice",
                "ns_education",
                "ns_environmental",
                "ns_health_social",
                "ns_municipal",
                "ns_other"
            ]
        }

        self._initialize_qdrant()
        self._initialize_openai()

    def _initialize_qdrant(self):
        """Initialize Qdrant connection."""
        try:
            qdrant_url = os.getenv('QDRANT_URL', 'https://qdrant-production-d86e.up.railway.app')

            # For Railway deployment, use HTTP-based client with increased timeouts
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                timeout=120,  # Increased timeout for Railway
                prefer_grpc=False,  # Use HTTP for Railway
                https=True,  # Railway uses HTTPS
                api_key=None,  # Self-hosted, no API key
                host=None,  # Use URL instead
                port=None   # Use URL instead
            )

            # Test connection with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    collections = self.qdrant_client.get_collections()
                    logger.info(f"✅ Qdrant initialized: {qdrant_url}")
                    logger.info(f"📊 Found {len(collections.collections)} collections")
                    break
                except Exception as retry_error:
                    if attempt == max_retries - 1:
                        raise retry_error
                    logger.warning(f"Qdrant connection attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            # Don't fail completely - service can still work for other functions
            self.qdrant_client = None

    def _initialize_openai(self):
        """Initialize OpenAI connection avoiding Railway proxy conflicts."""
        self.openai_client = None

        logger.info("🔧 Starting OpenAI initialization with proxy workaround...")

        try:
            # Check if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"🔧 API key environment variable present: {bool(api_key)}")

            if not api_key:
                logger.error("❌ OPENAI_API_KEY environment variable not found")
                return

            # Railway-specific workaround: Clear any proxy-related environment variables
            # that might interfere with OpenAI client initialization
            original_env = {}
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'proxies']

            for var in proxy_vars:
                if var in os.environ:
                    original_env[var] = os.environ[var]
                    del os.environ[var]
                    logger.info(f"🔧 Temporarily removed {var} environment variable")

            try:
                from openai import OpenAI
                logger.info("🔧 Creating OpenAI client with clean environment...")

                # Initialize with minimal, explicit parameters
                self.openai_client = OpenAI(
                    api_key=api_key,
                    timeout=60.0,  # Explicit timeout
                    max_retries=2   # Explicit retries
                )

                # Test that it has required methods
                if hasattr(self.openai_client, 'embeddings') and hasattr(self.openai_client, 'chat'):
                    logger.info("✅ OpenAI client initialized successfully with proxy workaround")
                    logger.info(f"🔧 Client type: {type(self.openai_client)}")
                else:
                    logger.error("❌ OpenAI client missing expected methods")
                    self.openai_client = None

            finally:
                # Restore original environment variables
                for var, value in original_env.items():
                    os.environ[var] = value
                    logger.info(f"🔧 Restored {var} environment variable")

        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.openai_client = None

    def is_connected(self) -> bool:
        """Check if service is properly connected."""
        return self.qdrant_client is not None and self.openai_client is not None

    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using OpenAI."""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")

        try:
            loop = asyncio.get_event_loop()

            # Use OpenAI v1+ API pattern - following official documentation
            # Using text-embedding-3-small to match the database embeddings
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query]  # input should be a list according to v1+ API
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
        """Search for relevant legal documents in Qdrant."""
        if not self.qdrant_client:
            raise Exception("Qdrant client not initialized")

        try:
            # Get query embedding
            query_embedding = await self.get_query_embedding(query)

            all_results = []

            # Search each jurisdiction's collections
            for jurisdiction in jurisdictions:
                if jurisdiction not in self.collection_mapping:
                    logger.warning(f"Unknown jurisdiction: {jurisdiction}")
                    continue

                collections = self.collection_mapping[jurisdiction]
                results_per_collection = max(1, top_k // len(collections))

                # Search across all collections for this jurisdiction
                for collection_name in collections:
                    try:
                        # Perform vector search
                        loop = asyncio.get_event_loop()
                        search_response = await loop.run_in_executor(
                            None,
                            lambda: self.qdrant_client.search(
                                collection_name=collection_name,
                                query_vector=query_embedding,
                                limit=results_per_collection
                            )
                        )

                        # Process results
                        for match in search_response:
                            result = SearchResult(
                                content=self._extract_content(match.payload),
                                metadata=match.payload,
                                similarity_score=float(match.score),
                                jurisdiction=jurisdiction,
                                chunk_id=str(match.id)
                            )
                            all_results.append(result)

                    except Exception as e:
                        logger.warning(f"Search failed for collection {collection_name}: {e}")
                        continue

            # Sort by similarity score (higher is better in Qdrant)
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)

            # Return top results across all jurisdictions
            return all_results[:top_k * len(jurisdictions)]

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _extract_content(self, payload: Dict[str, Any]) -> str:
        """Extract content from Qdrant payload."""
        # Try different possible content fields based on our data structure
        content_fields = ['content', 'text', 'content_preview', 'description']

        for field in content_fields:
            if field in payload and payload[field]:
                return str(payload[field])

        # Fallback: create content from available metadata
        title = payload.get('title', payload.get('document_title', 'Unknown Document'))
        section = payload.get('section', payload.get('section_label', ''))
        doc_type = payload.get('document_type', '')

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
                # Format based on jurisdiction for better context
                if result.jurisdiction == "federal":
                    doc_title = result.metadata.get('title', 'Unknown Document')
                    section_info = result.metadata.get('section_label', '')
                    if section_info:
                        section_info = f"Section {section_info}"
                else:  # nova_scotia
                    doc_title = result.metadata.get('title', result.metadata.get('document_title', 'Unknown Document'))
                    section_info = result.metadata.get('section', result.metadata.get('section_label', ''))
                    if section_info:
                        section_info = f"Section {section_info}"

                result_text = f"""
Jurisdiction: {result.jurisdiction.title()}
Document: {doc_title}
{f"Section: {section_info}" if section_info else ""}
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

            # Use OpenAI v1+ API pattern - following official documentation
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
        """Get database statistics from Qdrant."""
        if not self.qdrant_client:
            raise Exception("Qdrant client not initialized")

        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                None,
                lambda: self.qdrant_client.get_collections()
            )

            collection_stats = {}
            total_vectors = 0

            for collection in collections.collections:
                try:
                    collection_info = await loop.run_in_executor(
                        None,
                        lambda: self.qdrant_client.get_collection(collection.name)
                    )

                    points_count = collection_info.points_count
                    collection_stats[collection.name] = {
                        "vector_count": points_count,
                        "status": "green"
                    }
                    total_vectors += points_count

                except Exception as e:
                    logger.warning(f"Could not get stats for {collection.name}: {e}")
                    collection_stats[collection.name] = {
                        "vector_count": 0,
                        "status": "error"
                    }

            # Group by jurisdiction for cleaner output
            jurisdictions = {}
            for jurisdiction, collections in self.collection_mapping.items():
                jurisdictions[jurisdiction] = {
                    "vector_count": sum(
                        collection_stats.get(col, {}).get("vector_count", 0)
                        for col in collections
                    ),
                    "collections": {
                        col: collection_stats.get(col, {"vector_count": 0})
                        for col in collections
                    }
                }

            return {
                "total_vectors": total_vectors,
                "dimension": 1536,  # text-embedding-3-small dimension
                "jurisdictions": jurisdictions,
                "total_collections": len(collection_stats)
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise