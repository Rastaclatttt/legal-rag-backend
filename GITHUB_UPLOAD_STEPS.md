# GitHub Upload Steps

## Backend Ready for Upload! ✅

Your legal RAG backend is complete and ready to upload to GitHub. Since Git authentication requires manual setup, please follow these steps:

### 1. Upload to GitHub Repository

You have already created the GitHub repository at: `https://github.com/Rastaclatttt/legal-rag-backend.git`

**Option A: GitHub Web Interface (Easiest)**
1. Go to https://github.com/Rastaclatttt/legal-rag-backend
2. Click "uploading an existing file" or "Add file" > "Upload files"
3. Drag and drop ALL files from `/Users/lgraham/Downloads/consolidated-statutes-xml/legal-rag-backend/`
4. Commit with message: "Initial upload of legal RAG backend"

**Option B: Command Line (if you have Git configured)**
```bash
cd /Users/lgraham/Downloads/consolidated-statutes-xml/legal-rag-backend
git push -u origin main
```

### 2. Files to Upload
Make sure these files are uploaded:
- ✅ `pinecone_backend.py` (main FastAPI application)
- ✅ `main.py` (alternative backend without Pinecone)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `Procfile` (Railway deployment config)
- ✅ `railway.toml` (Railway configuration)
- ✅ `DEPLOY.md` (deployment guide)
- ✅ `.gitignore` (Git ignore file)

### 3. Deploy to Railway

Once uploaded to GitHub:

1. **Go to Railway**: https://railway.app
2. **New Project** > "Deploy from GitHub repo"
3. **Select Repository**: `Rastaclatttt/legal-rag-backend`
4. **Set Environment Variables**:
   ```
   OPENAI_API_KEY=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
   PINECONE_API_KEY=pcsk_4Bxjm8_BTwZfoEYrMhxnvnEZ8WQA6Peq5UY99FGmWqLP64EMDsy8DF1isy2RrQUFWQE8Qf
   PINECONE_INDEX_NAME=legal-rag-index
   PORT=8000
   ```
5. **Deploy** - Railway will automatically detect and deploy your FastAPI app

### 4. Update Frontend

Once deployed, Railway will give you a URL like: `https://your-app-name.railway.app`

Update your Vercel frontend environment variable:
- `NEXT_PUBLIC_API_URL` = your Railway backend URL

### Status
- ✅ Backend code ready
- ✅ GitHub repository created
- ⏳ Manual upload required
- ⏳ Railway deployment pending
- ⏳ Frontend URL update pending

The backend includes Pinecone integration with local fallback, so it will work regardless of Pinecone connectivity status.