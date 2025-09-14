#!/bin/bash
# Railway Environment Variables Setup Script

echo "🚂 Railway Environment Variables Setup"
echo "=========================================="

echo ""
echo "Step 1: Login to Railway"
echo "------------------------"
echo "Run: railway login"
echo "(This will open your browser to authenticate)"
echo ""

echo "Step 2: Link to your project"
echo "----------------------------"
echo "Run: railway link"
echo "(Select your legal-rag-backend-production project)"
echo ""

echo "Step 3: Set Environment Variables"
echo "---------------------------------"
echo "Run these commands:"
echo ""
echo "# Set Qdrant URL"
echo "railway variables set QDRANT_URL=https://qdrant-production-d86e.up.railway.app"
echo ""
echo "# Set OpenAI API Key (replace with your actual key)"
echo "railway variables set OPENAI_API_KEY=your-openai-api-key-here"
echo ""

echo "Step 4: Verify Variables"
echo "-----------------------"
echo "railway variables"
echo ""

echo "Step 5: Test Backend"
echo "-------------------"
echo "curl https://legal-rag-backend-production.up.railway.app/health"
echo ""

echo "🎯 Expected Result:"
echo '{"status":"healthy","service":"Legal RAG API","version":"1.0.0","qdrant_connected":true}'