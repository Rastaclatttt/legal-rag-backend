#!/usr/bin/env python3
"""
Simplified OpenAI test for Railway environment
"""
import os

def main():
    print("=== Simple OpenAI Test ===")

    # Test 1: Environment
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"OPENAI_API_KEY present: {bool(api_key)}")

    if not api_key:
        print("❌ No API key found")
        return

    # Test 2: Import
    try:
        from openai import OpenAI
        print("✅ OpenAI imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return

    # Test 3: Client creation
    try:
        client = OpenAI()  # Should use env var automatically
        print("✅ Client created successfully")
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return

    # Test 4: Simple API call
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=["hello world"]
        )
        print(f"✅ API call successful, dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"❌ API call failed: {e}")

if __name__ == "__main__":
    main()