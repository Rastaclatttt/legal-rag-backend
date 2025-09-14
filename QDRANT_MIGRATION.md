# Qdrant Migration Guide

## Overview

Your backend has been successfully migrated from Pinecone to Qdrant. Here's what changed:

## Changes Made

### 1. New Service File
- **Created**: `qdrant_rag_service.py` - New RAG service using Qdrant
- **Replaces**: `rag_service.py` - Old Pinecone-based service

### 2. Updated Dependencies
- **Removed**: `pinecone-client==2.2.4`
- **Added**: `qdrant-client==1.7.0`

### 3. Updated App Configuration
- Modified `app.py` to use `qdrant_rag_service` instead of `rag_service`
- Updated health check to show `qdrant_connected` instead of `pinecone_connected`

### 4. Collection Mapping
The new service maps jurisdictions to your Qdrant collections:

**Federal**: (when federal data is available)
- `federal_statutes`
- `federal_regulations`
- `federal_acts_a_c`, `federal_acts_d_h`, `federal_acts_i_m`, `federal_acts_n_s`, `federal_acts_t_z`
- `federal_other`

**Nova Scotia**: (currently available)
- `ns_administrative` (9,351 points)
- `ns_business_commerce` (6,351 points)
- `ns_criminal_justice` (6,140 points)
- `ns_education` (2,626 points)
- `ns_environmental` (1,328 points)
- `ns_health_social` (2,633 points)
- `ns_municipal` (3,139 points)
- `ns_other` (53,148 points)

## Environment Variables

### Required
- `OPENAI_API_KEY` - Your OpenAI API key
- `QDRANT_URL` - Your Railway Qdrant URL (default: `https://qdrant-production-d86e.up.railway.app`)

### No Longer Needed
- `PINECONE_API_KEY` - Can be removed
- `PINECONE_INDEX_NAME` - Can be removed

## Deployment

1. Install new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export QDRANT_URL="https://qdrant-production-d86e.up.railway.app"
   export OPENAI_API_KEY="your-openai-key"
   ```

3. Run the backend:
   ```bash
   python app.py
   ```

## API Endpoints (Unchanged)

All API endpoints remain the same:
- `GET /health` - Health check
- `POST /search` - Vector search only
- `POST /rag` - Full RAG with AI response
- `GET /jurisdictions` - Available jurisdictions
- `GET /stats` - Database statistics
- `GET /docs` - API documentation

## Benefits of Qdrant Migration

1. **Cost Reduction**: No Pinecone subscription fees
2. **Self-Hosted Control**: Full control over your vector database
3. **Railway Integration**: Optimized for Railway deployment
4. **Better Organization**: Collections organized by legal topic
5. **Improved Performance**: Direct HTTP connections to Railway Qdrant

## Testing

Test your migrated backend:

```bash
# Health check
curl http://localhost:8000/health

# Search test
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "incorporation requirements", "jurisdictions": ["nova_scotia"]}'

# RAG test
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for business registration?", "jurisdictions": ["nova_scotia"]}'
```

## Rollback (if needed)

If you need to rollback to Pinecone:
1. Change import in `app.py` back to `from rag_service import LegalRAGService`
2. Restore old `requirements.txt` with `pinecone-client`
3. Set Pinecone environment variables

## Support

Your Qdrant instance is running at: `https://qdrant-production-d86e.up.railway.app`

- **Total Documents**: 84,716 Nova Scotia legal documents
- **Collections**: 8 organized by legal topic
- **Vector Dimension**: 1536 (OpenAI ada-002)
- **Storage**: Persistent Railway volume at `/qdrant/storage`