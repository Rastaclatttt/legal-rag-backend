#!/usr/bin/env python3
"""
Test Qdrant connection and import first 100 NS statutes
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json
import os
import uuid

def main():
    """Test connection and import sample data"""
    
    print("🔍 TESTING QDRANT CONNECTION AND SAMPLE IMPORT")
    print("=" * 60)
    
    qdrant_url = os.getenv('QDRANT_URL', 'https://qdrant-production-d86e.up.railway.app')
    print(f"🌐 Connecting to: {qdrant_url}")
    
    # Create client
    client = QdrantClient(
        url=qdrant_url,
        timeout=120,
        prefer_grpc=False,
        https=True,
        api_key=None,
    )
    
    # Test connection
    try:
        collections = client.get_collections()
        print(f"✅ Connected! Current collections: {len(collections.collections)}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Load first 100 NS statutes
    ns_file = "../NS_Consolidated_Statutes/ns_statutes_embeddings.json"
    
    if not os.path.exists(ns_file):
        print(f"❌ File not found: {ns_file}")
        return
    
    print(f"📂 Loading first 100 entries from: {ns_file}")
    
    # Read just first 100 entries to test
    print("📂 Loading sample data...")
    with open(ns_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_sample = data[:100]  # First 100 items
    
    if not data_sample:
        print("❌ No data found")
        return
    
    print(f"📊 Loaded {len(data_sample)} sample entries")
    print(f"🔢 Vector dimension: {len(data_sample[0]['embedding'])}")
    
    # Create collection
    collection_name = "nova_scotia_statutes"
    vector_size = len(data_sample[0]['embedding'])
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {collection_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️ Collection {collection_name} already exists")
        else:
            print(f"❌ Failed to create collection: {e}")
            return
    
    # Import sample points
    points = []
    for i, item in enumerate(data_sample):
        points.append(PointStruct(
            id=item.get('text_id', str(uuid.uuid4())),
            vector=item['embedding'],
            payload={
                'text': item['text'],
                'text_id': item.get('text_id'),
                'jurisdiction': 'nova_scotia',
                'document_type': 'statute'
            }
        ))
    
    try:
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Imported {len(points)} sample points successfully!")
        
        # Verify
        info = client.get_collection(collection_name)
        print(f"📋 Collection now has {info.points_count} total points")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")

if __name__ == "__main__":
    main()