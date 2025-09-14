# Vercel Deployment Guide for Legal RAG Backend

## Quick Setup

### 1. Deploy Backend to Vercel

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy from the backend directory**:
   ```bash
   cd legal-rag-backend
   vercel --prod
   ```

4. **Set Environment Variables in Vercel Dashboard**:
   - Go to your Vercel project dashboard
   - Navigate to Settings > Environment Variables
   - Add these variables:
     ```
     OPENAI_API_KEY=your-openai-api-key
     QDRANT_URL=https://qdrant-production-d86e.up.railway.app
     ```

### 2. Update Frontend Configuration

After deployment, update your frontend to use the Vercel backend URL:

1. Get your Vercel backend URL (e.g., `https://your-backend.vercel.app`)
2. Update the frontend environment variable or hardcode the URL

## Files Created/Modified

### ✅ vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.9"
  }
}
```

### ✅ app.py modifications
- Added `handler = app` for Vercel compatibility
- Updated CORS to allow Vercel domains

### ✅ Environment Variables Required
- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: Your Railway Qdrant URL (https://qdrant-production-d86e.up.railway.app)

## Testing Your Deployment

After deployment, test these endpoints:

```bash
# Replace YOUR_BACKEND_URL with your actual Vercel URL
export BACKEND_URL="https://your-backend.vercel.app"

# Health check
curl $BACKEND_URL/health

# Search test
curl -X POST $BACKEND_URL/search \
  -H "Content-Type: application/json" \
  -d '{"query": "incorporation requirements", "jurisdictions": ["nova_scotia"]}'

# RAG test
curl -X POST $BACKEND_URL/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for business registration?", "jurisdictions": ["nova_scotia"]}'
```

## Frontend Integration

Update your frontend API base URL to point to your Vercel backend:

```typescript
// In your frontend app/page.tsx
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://your-backend.vercel.app'
```

## Deployment Commands

```bash
# One-time setup
cd legal-rag-backend
vercel login

# Deploy
vercel --prod

# Check deployment status
vercel ls

# View logs
vercel logs
```

## Architecture Overview

```
Frontend (Vercel) → Backend (Vercel) → Qdrant (Railway)
     ↓                    ↓                  ↓
  Next.js            FastAPI           Vector Database
  React              Python            84,716 NS docs
  TypeScript         OpenAI            8 collections
```

## Environment Variables Setup

In Vercel Dashboard > Settings > Environment Variables:

| Variable | Value | Environment |
|----------|-------|-------------|
| `OPENAI_API_KEY` | `sk-...` | Production |
| `QDRANT_URL` | `https://qdrant-production-d86e.up.railway.app` | Production |

## Troubleshooting

### Common Issues:

1. **Build fails**: Check Python version compatibility
2. **Timeout errors**: Qdrant connection issues
3. **CORS errors**: Update frontend domain in CORS settings
4. **Cold starts**: First request may be slow

### Solutions:

1. **Python Version**: Vercel supports Python 3.9 (specified in vercel.json)
2. **Timeouts**: Increase timeout in qdrant_rag_service.py
3. **CORS**: Add your frontend domain to allowed origins
4. **Cold Starts**: Consider using Vercel Pro for better performance

## Next Steps

1. Deploy backend to Vercel
2. Set environment variables
3. Test API endpoints
4. Update frontend API URL
5. Deploy frontend to Vercel
6. Test full integration