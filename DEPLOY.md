# Railway Deployment Guide

## Quick Deploy

The backend is ready for Railway deployment. Follow these steps:

### 1. Railway Setup
1. Go to [railway.app](https://railway.app) and sign up/login
2. Click "New Project" 
3. Choose "Deploy from GitHub repo"
4. Connect your GitHub account and select this repository
5. Choose the `legal-rag-backend` folder as the root directory

### 2. Environment Variables
Set these environment variables in Railway:

```
OPENAI_API_KEY=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
PINECONE_API_KEY=pcsk_4Bxjm8_BTwZfoEYrMhxnvnEZ8WQA6Peq5UY99FGmWqLP64EMDsy8DF1isy2RrQUFWQE8Qf
PINECONE_INDEX_NAME=legal-rag-index
PORT=8000
```

### 3. Deploy
- Railway will automatically detect the Python app and deploy it
- The `Procfile` specifies the startup command
- The app will be available at the Railway-provided URL

### 4. Update Frontend
Once deployed, update the `NEXT_PUBLIC_API_URL` in your Vercel frontend to point to the Railway backend URL.

## Local Testing
Run locally with:
```bash
export OPENAI_API_KEY='your-key'
export PINECONE_API_KEY='your-key'  
export PINECONE_INDEX_NAME='legal-rag-index'
uvicorn pinecone_backend:app --reload
```

## Features
- ✅ FastAPI backend with Pinecone integration
- ✅ Local vector store fallback
- ✅ Health check endpoint at `/health`
- ✅ RAG endpoint at `/rag`
- ✅ CORS enabled for frontend integration
- ✅ Proper error handling and logging