#!/usr/bin/env python3
"""
Database Population Script for Legal RAG
Populates Railway Qdrant with federal and Nova Scotia legal documents
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json
import os
from tqdm import tqdm
import time
import uuid

def create_robust_client(url):
    """Create Qdrant client with Railway-optimized settings"""
    return QdrantClient(
        url=url,
        timeout=120,  # Increase timeout further for Railway
        prefer_grpc=False,  # Use HTTP for Railway
        https=True,  # Railway uses HTTPS
        api_key=None,  # No API key for self-hosted
    )

def import_ns_statutes(client, json_file: str, batch_size: int = 50):
    """Import Nova Scotia statutes from single embeddings file"""

    print(f"📦 Importing Nova Scotia statutes from {json_file}...")

    # Load data
    print("📂 Loading Nova Scotia embeddings file...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check data structure and extract points
    if isinstance(data, list):
        points_data = data
        vector_size = len(data[0]['embedding']) if data else 1536
    elif isinstance(data, dict) and 'points' in data:
        points_data = data['points']
        vector_size = data.get('vector_size', 1536)
    else:
        # Assume it's a flat structure with embeddings
        points_data = data
        vector_size = 1536

    print(f"📊 Found {len(points_data)} Nova Scotia statute chunks...")
    print(f"🔢 Vector dimension: {vector_size}")

    # Create a single collection for all NS statutes
    collection_name = "nova_scotia_statutes"

    # Create collection with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"✅ Created collection: {collection_name}")
            break
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ Collection {collection_name} already exists")
                break
            elif attempt < max_retries - 1:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
            else:
                print(f"❌ Failed to create collection after {max_retries} attempts: {e}")
                return False

    # Import points in batches
    total_points = len(points_data)
    successful_imports = 0

    for i in tqdm(range(0, total_points, batch_size), desc=f"Importing NS Statutes"):
        batch = points_data[i:i + batch_size]

        # Convert to PointStruct
        points = []
        for j, point_data in enumerate(batch):
            # Handle different data formats
            if 'id' in point_data:
                point_id = point_data['id']
            else:
                point_id = str(uuid.uuid4())

            if 'vector' in point_data:
                vector = point_data['vector']
            elif 'embedding' in point_data:
                vector = point_data['embedding']
            else:
                print(f"⚠️ No vector found in point {j}, skipping")
                continue

            if 'payload' in point_data:
                payload = point_data['payload']
            else:
                # Create payload from available data
                payload = {k: v for k, v in point_data.items()
                          if k not in ['id', 'vector', 'embedding']}

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        if not points:
            continue

        # Upload batch with retries
        for attempt in range(3):
            try:
                client.upsert(collection_name=collection_name, points=points)
                successful_imports += len(points)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"⚠️ Batch {i//batch_size + 1} attempt {attempt + 1} failed: {e}")
                    time.sleep(2)
                else:
                    print(f"❌ Batch {i//batch_size + 1} failed permanently: {e}")

    print(f"✅ Imported {successful_imports}/{total_points} points to {collection_name}")
    return successful_imports == total_points

def main():
    """Import Nova Scotia statutes to Railway Qdrant"""

    print("🚀 NOVA SCOTIA STATUTES IMPORT")
    print("=" * 50)

    qdrant_url = os.getenv('QDRANT_URL', 'https://qdrant-production-d86e.up.railway.app')
    print(f"🌐 Connecting to: {qdrant_url}")

    # Create client with retries
    client = None
    for attempt in range(5):
        try:
            client = create_robust_client(qdrant_url)

            # Test connection with timeout
            print(f"🔄 Connection attempt {attempt + 1}/5...")
            collections = client.get_collections()
            print(f"✅ Connected! Current collections: {len(collections.collections)}")
            break

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < 4:
                print(f"⏳ Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"❌ Failed to connect after 5 attempts: {e}")
                return

    # Path to Nova Scotia embeddings file
    ns_embeddings_file = "../NS_Consolidated_Statutes/ns_statutes_embeddings.json"

    if not os.path.exists(ns_embeddings_file):
        print(f"❌ Nova Scotia embeddings file not found: {ns_embeddings_file}")
        return

    print(f"📁 Found Nova Scotia embeddings: {ns_embeddings_file}")

    # Import Nova Scotia statutes
    print("\n🏛️ Importing Nova Scotia Statutes...")
    success = import_ns_statutes(client, ns_embeddings_file, batch_size=25)

    print(f"\n📊 Import Summary:")
    print(f"{'✅ SUCCESS' if success else '❌ FAILED'}: Nova Scotia Statutes")

    # Show final collection info
    try:
        collections = client.get_collections()
        print(f"\n📁 Total collections in Qdrant: {len(collections.collections)}")

        for collection in collections.collections:
            try:
                info = client.get_collection(collection.name)
                print(f"  📋 {collection.name}: {info.points_count} points")
            except Exception as e:
                print(f"  📋 {collection.name}: info unavailable ({e})")

    except Exception as e:
        print(f"⚠️ Could not get collection info: {e}")

    print("\n🎉 Nova Scotia statutes import complete!")

if __name__ == "__main__":
    main()